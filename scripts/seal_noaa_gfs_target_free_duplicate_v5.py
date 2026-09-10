"""Seal the V5 target-free duplicate experiment without interpreting model data.

This module has two deliberately separate roles:

* code-seal mode validates the frozen V3, V4, incident, and V5 documentary
  lineage, the exact six V5 code/test identities, and externally supplied
  synthetic isolated-test evidence, then publishes only the V5 code seal;
* final mode validates the complete authority chain and a no-fit postrun pass,
  then publishes the already audited staged outputs in the frozen order with
  ``TRACK_A_TARGET_FREE_RUN_MANIFEST.json`` last.

The module uses the standard library for every operation except one narrowly
scoped, post-publication PyArrow footer check on the provenance alias.  It
never calls a Parquet table/row-group/value reader, imports pandas, NumPy,
SciPy, sklearn, joblib, or a network client, and has no model-fit route.  The
only final-name publication primitive is a create-if-absent hard link.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT_DEFAULT = (
    REPO / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
)

AMENDMENT_RELATIVE = "prereg/target_free_multiseason_amendment_v3.json"
AMENDMENT_SIZE_BYTES = 102_361
AMENDMENT_SHA256 = "f9d072fc24ab29608efc6483098ada5a8d5e8129e1d1bd478e1d84438a2e9c14"
AMENDMENT_CANONICAL_SIZE_BYTES = 84_601
AMENDMENT_CANONICAL_SHA256 = (
    "ae5c5d05e63169112799c48a8ede52e8ce29319bc5d9e96927175ccbfd9b91d1"
)
AMENDMENT_V4_RELATIVE = "prereg/target_free_multiseason_amendment_v4.json"
AMENDMENT_V4_SIZE_BYTES = 14_385
AMENDMENT_V4_SHA256 = (
    "9f8abbe878bd10e7b91f78a03f44389a4973c8c1cb7da3c94752487957be7980"
)
AMENDMENT_V4_CANONICAL_SIZE_BYTES = 12_501
AMENDMENT_V4_CANONICAL_SHA256 = (
    "3ad1f38508b373aac70ee19b4253e91b9104f9527cb77952121757226f35abed"
)
AMENDMENT_V4_CREATED_UTC = "2026-08-11T15:00:30.854000Z"

INCIDENT_RELATIVE = (
    "incidents/TARGET_FREE_MULTISEASON_V3_EXECUTION_OR_PUBLICATION_FAILURE_V1.json"
)
INCIDENT_SIZE_BYTES = 44_709
INCIDENT_SHA256 = "b5dc9dc117f8686bb7d80f73e082c895c55af832e8b25d33d50661cb19329759"
INCIDENT_CANONICAL_SIZE_BYTES = 35_456
INCIDENT_CANONICAL_SHA256 = (
    "6d2cc2593ea4375ddb88e6803057595ea34c04eac04be917a07c185eac2bec7a"
)
INCIDENT_CREATED_UTC = "2026-08-11T17:31:31.631388Z"

AMENDMENT_V5_RELATIVE = "prereg/target_free_multiseason_amendment_v5.json"
AMENDMENT_V5_SIZE_BYTES = 59_798
AMENDMENT_V5_SHA256 = "3e5515f275cd572def713d640aee9d5ecedb982f98d4983ee2d8adcdff0c5d9b"
AMENDMENT_V5_CANONICAL_SIZE_BYTES = 48_167
AMENDMENT_V5_CANONICAL_SHA256 = (
    "7533dd4ef5e4be293109409ed17730ce10a855dfb8da7f9f0a66dd8e8ef9371f"
)
AMENDMENT_V5_CREATED_UTC = "2026-08-11T18:03:24.805000Z"

CODE_SEAL_RELATIVE = "prereg/target_free_duplicate_code_seal_v5.json"
AUTHORIZATION_RELATIVE = "prereg/target_free_duplicate_execution_authorization_v5.json"
REVIEW_RELATIVE = (
    "independent_redteam/TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_REVIEW_V5.json"
)
GO_RELATIVE = "independent_redteam/TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_GO_V5.json"
POSTRUN_RELATIVE = "independent_redteam/TRACK_A_TARGET_FREE_DUPLICATE_POSTRUN_PASS_V5.json"

OUTPUT_ROOT_RELATIVE = "modeling/target_free_multiseason_v5/"
TRANSACTION_ROOT_RELATIVE = "modeling/target_free_multiseason_v5/.transaction_v5/"

CODE_ROLE_PATHS = {
    "runner": "scripts/run_noaa_gfs_target_free_duplicate_v5.py",
    "runner_test": "tests/test_noaa_gfs_target_free_duplicate_v5.py",
    "auditor": "scripts/audit_noaa_gfs_target_free_duplicate_v5.py",
    "auditor_test": "tests/test_audit_noaa_gfs_target_free_duplicate_v5.py",
    "sealer": "scripts/seal_noaa_gfs_target_free_duplicate_v5.py",
    "sealer_test": "tests/test_seal_noaa_gfs_target_free_duplicate_v5.py",
}
V3_CODE_ROLE_PATHS = {
    "runner": "scripts/run_noaa_gfs_target_free_duplicate_v3.py",
    "runner_test": "tests/test_noaa_gfs_target_free_duplicate_v3.py",
    "auditor": "scripts/audit_noaa_gfs_target_free_duplicate_v3.py",
    "auditor_test": "tests/test_audit_noaa_gfs_target_free_duplicate_v3.py",
    "sealer": "scripts/seal_noaa_gfs_target_free_duplicate_v3.py",
    "sealer_test": "tests/test_seal_noaa_gfs_target_free_duplicate_v3.py",
}
TEST_ROLE_PATHS = {
    "runner": CODE_ROLE_PATHS["runner_test"],
    "auditor": CODE_ROLE_PATHS["auditor_test"],
    "sealer": CODE_ROLE_PATHS["sealer_test"],
}

OUTPUT_NAMESPACE = {
    "root": OUTPUT_ROOT_RELATIVE,
    "runner_transaction_root": TRANSACTION_ROOT_RELATIVE,
    "require_absent_at_go": True,
    "single_attempt_only": True,
}

ALIAS_SPECS = {
    "FIELD_CENSUS_LOCK.json": {
        "source": "manifest_census_v1.json",
        "stage": "modeling/target_free_multiseason_v5/.alias_stage_FIELD_CENSUS_LOCK_v5",
        "final": "modeling/target_free_multiseason_v5/FIELD_CENSUS_LOCK.json",
        "size_bytes": 4_523,
        "sha256": "0269f6717b1110fafb6bba052c78824c616c648dbbdfe9f630947012c3f571a1",
        "bundle": (
            "manifest_census_v1.json",
            "census/RAW_DOWNLOAD_PLAN_EXACT.json",
            "independent_redteam/TRACK_A_CENSUS_INDEPENDENT_AUDIT_V1.json",
            "audit/CENSUS_CUMULATIVE_ACCESS_AMENDMENT.json",
        ),
    },
    "PROVENANCE_LEDGER.parquet": {
        "source": "raw/RAW_RANGE_MANIFEST.parquet",
        "stage": "modeling/target_free_multiseason_v5/.alias_stage_PROVENANCE_LEDGER_v5",
        "final": "modeling/target_free_multiseason_v5/PROVENANCE_LEDGER.parquet",
        "size_bytes": 3_812_043,
        "sha256": "11849682fd1c65ca20cfcb3002d3708beccf10ef862bb0f03d054c2c030790d5",
        "bundle": (
            "raw/RAW_RANGE_MANIFEST.parquet",
            "raw/RAW_RANGE_MANIFEST.csv",
            "raw/RAW_ACCESS_LEDGER.json",
            "decoded/DECODED_MATRIX_LOCK.json",
            "decoded/PHYSICAL_AND_COVERAGE_AUDIT.json",
            "independent_redteam/V8_LIVE_AUDIT_SUCCESS_ATTESTATION_V1.json",
        ),
    },
}
ALIAS_ORDER = tuple(ALIAS_SPECS)

# Immutable historical V3 literals are validated independently.  They must
# never be synthesized by a global V3-to-V5 replacement.
V3_OUTPUT_ROOT_RELATIVE = "modeling/target_free_multiseason_v3/"
V3_TRANSACTION_ROOT_RELATIVE = "modeling/target_free_multiseason_v3/.transaction_v3/"
V3_ALIAS_SPECS = {
    "FIELD_CENSUS_LOCK.json": {
        **ALIAS_SPECS["FIELD_CENSUS_LOCK.json"],
        "stage": "modeling/target_free_multiseason_v3/.alias_stage_FIELD_CENSUS_LOCK_v3",
        "final": "modeling/target_free_multiseason_v3/FIELD_CENSUS_LOCK.json",
    },
    "PROVENANCE_LEDGER.parquet": {
        **ALIAS_SPECS["PROVENANCE_LEDGER.parquet"],
        "stage": "modeling/target_free_multiseason_v3/.alias_stage_PROVENANCE_LEDGER_v3",
        "final": "modeling/target_free_multiseason_v3/PROVENANCE_LEDGER.parquet",
    },
}
V3_RUNNER_PUBLICATION_ORDER = (
    "TARGET_FREE_DUPLICATE_METRICS.csv",
    "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv",
    "TARGET_FREE_FAMILY_DECISION.json",
    "TARGET_FREE_DUPLICATE_OOF_V3.parquet",
    "TARGET_FREE_FIT_LEDGER_V3.json",
    "TARGET_FREE_DISTRIBUTION_STABILITY_V3.json",
    "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3.parquet",
)
V3_FINAL_PUBLICATION_ORDER = V3_RUNNER_PUBLICATION_ORDER + (
    "TARGET_FREE_FAMILY_LOCK.json",
    "TRACK_A_TARGET_FREE_RUN_MANIFEST.json",
)
V3_EXACT11_SCHEMA_ORDER = ALIAS_ORDER + (
    "TARGET_FREE_DUPLICATE_METRICS.csv",
    "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv",
    "TARGET_FREE_FAMILY_DECISION.json",
    "TARGET_FREE_FAMILY_LOCK.json",
    "TARGET_FREE_DUPLICATE_OOF_V3.parquet",
    "TARGET_FREE_FIT_LEDGER_V3.json",
    "TARGET_FREE_DISTRIBUTION_STABILITY_V3.json",
    "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3.parquet",
    "TRACK_A_TARGET_FREE_RUN_MANIFEST.json",
)
V3_RUNNER_STAGED_RELATIVES = {
    name: f"{V3_TRANSACTION_ROOT_RELATIVE}{name}"
    for name in V3_RUNNER_PUBLICATION_ORDER
}
V3_SEALER_STAGED_RELATIVES = {
    "TARGET_FREE_FAMILY_LOCK.json": (
        f"{V3_TRANSACTION_ROOT_RELATIVE}TARGET_FREE_FAMILY_LOCK.json"
    ),
    "TRACK_A_TARGET_FREE_RUN_MANIFEST.json": (
        f"{V3_TRANSACTION_ROOT_RELATIVE}TRACK_A_TARGET_FREE_RUN_MANIFEST.json"
    ),
}

RUNNER_PUBLICATION_ORDER = (
    "TARGET_FREE_DUPLICATE_METRICS.csv",
    "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv",
    "TARGET_FREE_FAMILY_DECISION.json",
    "TARGET_FREE_DUPLICATE_OOF_V5.parquet",
    "TARGET_FREE_FIT_LEDGER_V5.json",
    "TARGET_FREE_DISTRIBUTION_STABILITY_V5.json",
    "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V5.parquet",
)
FINAL_PUBLICATION_ORDER = RUNNER_PUBLICATION_ORDER + (
    "TARGET_FREE_FAMILY_LOCK.json",
    "TRACK_A_TARGET_FREE_RUN_MANIFEST.json",
)
EXACT11_BASENAMES = ALIAS_ORDER + FINAL_PUBLICATION_ORDER
# The amendment's complete-path inventory is an identity-bearing schema order,
# not the final publication order.  In particular, the family lock is listed
# before the seven large evidence outputs even though it is built and published
# by the sealer after those seven outputs have passed the postrun audit.
EXACT11_SCHEMA_ORDER = ALIAS_ORDER + (
    "TARGET_FREE_DUPLICATE_METRICS.csv",
    "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv",
    "TARGET_FREE_FAMILY_DECISION.json",
    "TARGET_FREE_FAMILY_LOCK.json",
    "TARGET_FREE_DUPLICATE_OOF_V5.parquet",
    "TARGET_FREE_FIT_LEDGER_V5.json",
    "TARGET_FREE_DISTRIBUTION_STABILITY_V5.json",
    "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V5.parquet",
    "TRACK_A_TARGET_FREE_RUN_MANIFEST.json",
)

RUNNER_STAGED_RELATIVES = {
    name: f"{TRANSACTION_ROOT_RELATIVE}{name}" for name in RUNNER_PUBLICATION_ORDER
}
SEALER_STAGED_RELATIVES = {
    "TARGET_FREE_FAMILY_LOCK.json": (
        f"{TRANSACTION_ROOT_RELATIVE}TARGET_FREE_FAMILY_LOCK.json"
    ),
    "TRACK_A_TARGET_FREE_RUN_MANIFEST.json": (
        f"{TRANSACTION_ROOT_RELATIVE}TRACK_A_TARGET_FREE_RUN_MANIFEST.json"
    ),
}
FINAL_RELATIVES = {
    name: f"{OUTPUT_ROOT_RELATIVE}{name}" for name in FINAL_PUBLICATION_ORDER
}

EXPECTED_RUNNER_ROWS = {
    "TARGET_FREE_DUPLICATE_METRICS.csv": 333,
    "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv": 348,
    "TARGET_FREE_FAMILY_DECISION.json": 1,
    "TARGET_FREE_DUPLICATE_OOF_V5.parquet": 352_512,
    "TARGET_FREE_FIT_LEDGER_V5.json": 1_332,
    "TARGET_FREE_DISTRIBUTION_STABILITY_V5.json": 624,
    "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V5.parquet": 414_720,
}
EXPECTED_RUNNER_FORMATS = {
    name: (
        "CSV" if name.endswith(".csv") else "PARQUET" if name.endswith(".parquet") else "JSON"
    )
    for name in RUNNER_PUBLICATION_ORDER
}

EXECUTION_BUDGET = {
    "attempt_count": 1,
    "alias_count": 2,
    "primary_model_fit_units": 72,
    "analytic_affine_fit_units": 1_260,
    "total_decision_fit_units": 1_332,
    "fit_ledger_rows": 1_332,
    "oof_rows": 352_512,
    "metrics_csv_rows": 333,
    "stability_csv_rows": 348,
    "distribution_records": 624,
    "endpoint_pooled_records": 3,
    "derived_rows": 414_720,
    "runner_staged_outputs": 7,
    "final_complete_outputs": 11,
}

DOCUMENTARY_SCOPE = {
    "documentary_only": True,
    "execution_authorized": False,
    "value_read_authorized": False,
    "fit_authorized": False,
    "output_publication_authorized": False,
    "network_authorized": False,
    "label_or_future_year_authorized": False,
    "submission_csv_authorized": False,
}
AUTHORIZATION_SCOPE = {
    "single_attempt_authorized": True,
    "alias_source_byte_copy_authorized": True,
    "alias_publication_authorized": True,
    "decoded_predictor_value_read_authorized": True,
    "primary_model_fit_units_authorized": 72,
    "analytic_affine_fit_units_authorized": 1_260,
    "staged_output_write_authorized": True,
    "postrun_auditor_authorized": True,
    "final_publication_conditioned_on_postrun_pass": True,
    "network_requests_allowed": 0,
    "labels_read_allowed": False,
    "arrays_2024_2025_read_allowed": False,
    "generation_model_fit_allowed": False,
    "submission_csv_allowed": False,
}
GO_SCOPE = {
    "single_attempt_go": True,
    "runner_go": True,
    "auditor_go": True,
    "sealer_go_conditioned_on_postrun_pass": True,
    "network_requests_allowed": 0,
    "labels_read_allowed": False,
    "arrays_2024_2025_read_allowed": False,
    "generation_model_fit_allowed": False,
    "submission_csv_allowed": False,
}
POSTRUN_SCOPE = {
    "documentary_only": True,
    "audit_pass_attested": True,
    "sealer_publication_condition_satisfied": True,
    "execution_authorized": False,
    "value_read_authorized": False,
    "fit_authorized": False,
    "network_authorized": False,
    "submission_csv_authorized": False,
}

FILE_IDENTITY_KEYS = frozenset({"path", "size_bytes", "sha256"})
AMENDMENT_IDENTITY_KEYS = frozenset(
    {"path", "size_bytes", "sha256", "canonical_size_bytes", "canonical_sha256"}
)
AMENDMENT_V4_TOP_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "status", "created_utc",
        "canonical_path", "bound_v3_amendment", "correction_scope",
        "pyarrow_22_0_0_evidence", "corrected_parquet_serialization",
        "corrected_csv_boundary_serialization", "affected_output_schemas_exact",
        "preserved_v3_semantics", "authority_scope",
        "required_future_control_bindings",
        "final_lock_manifest_binding_supersession", "required_tests",
        "prohibitions", "strict_schema",
    }
)
V4_CORRECTION_PATHS = (
    "/output_contract/output_serialization_exact/parquet",
    "/output_contract/output_serialization_exact/csv_boundary_flags",
)
V4_PARQUET_LITERAL = (
    "PYARROW_22_0_0_PARQUET_VERSION_2_6_ZSTD_LEVEL_3_USE_DICTIONARY_FALSE_"
    "WRITE_STATISTICS_TRUE_DATA_PAGE_VERSION_1_0_ROW_GROUP_SIZE_65536_INDEX_"
    "FALSE_COERCE_TIMESTAMPS_NONE_EFFECTIVE_NATIVE_NS_FOR_VERSION_2_6_ALLOW_"
    "TRUNCATED_TIMESTAMPS_FALSE_USE_DEPRECATED_INT96_FALSE_STORE_SCHEMA_TRUE_"
    "WRITE_PAGE_INDEX_FALSE_FLOAT_COLUMNS_BINARY64"
)
V4_CSV_BOUNDARY_LITERAL = (
    "CSV_BOOLEAN_CELLS_UPPERCASE_TRUE_OR_FALSE;BOUNDARY_EQUALITY_FLAGS_EMPTY_"
    "IF_NONE_ELSE_SEMICOLON_JOINED_ACTIVE_CONTINUOUS_CUTOFF_NAMES_IN_COMPONENT_"
    "DECISION_TRUTH_TABLE_CONTINUOUS_CUTOFFS_EXACT_ORDER_WITH_NO_DUPLICATES"
)
V4_EXPLICIT_WRITE_TABLE_KWARGS = {
    "allow_truncated_timestamps": False,
    "coerce_timestamps": None,
    "compression": "zstd",
    "compression_level": 3,
    "data_page_version": "1.0",
    "row_group_size": 65_536,
    "store_schema": True,
    "use_deprecated_int96_timestamps": False,
    "use_dictionary": False,
    "version": "2.6",
    "write_page_index": False,
    "write_statistics": True,
}
V4_UNLISTED_WRITE_TABLE_DEFAULTS = {
    "data_page_size": None,
    "flavor": None,
    "filesystem": None,
    "use_byte_stream_split": False,
    "column_encoding": None,
    "use_compliant_nested_type": True,
    "encryption_properties": None,
    "write_batch_size": None,
    "dictionary_pagesize_limit": None,
    "write_page_checksum": False,
    "sorting_columns": None,
    "store_decimal_as_integer": False,
}
OUTPUT_IDENTITY_KEYS = frozenset(
    {"path", "size_bytes", "sha256", "format", "row_count", "logical_sha256"}
)

CONTROL_SPECS = {
    "code_seal": {
        "keys": frozenset(
            {
                "schema_version", "artifact_type", "status", "created_utc",
                "amendment", "amendment_v4", "amendment_v5", "code_identities", "test_evidence", "output_namespace",
                "authority_scope", "required_next_controls", "publisher",
            }
        ),
        "artifact_type": "TARGET_FREE_DUPLICATE_CODE_SEAL_V5",
        "status": "SEALED_V5_CODE_AND_TESTS_NO_EXECUTION_AUTHORITY",
        "publisher": "V5_SEALER_CODE_SEAL_MODE_CREATE_IF_ABSENT_ONLY",
        "authority_scope": DOCUMENTARY_SCOPE,
    },
    "authorization": {
        "keys": frozenset(
            {
                "schema_version", "artifact_type", "status", "created_utc",
                "attempt_id", "amendment", "amendment_v4", "amendment_v5", "code_seal", "code_identities",
                "output_namespace", "execution_budget", "required_commands",
                "required_review", "required_go", "authority_scope", "publisher",
            }
        ),
        "artifact_type": "TARGET_FREE_DUPLICATE_EXECUTION_AUTHORIZATION_V5",
        "status": "AUTHORIZED_V5_SINGLE_ATTEMPT_PENDING_INDEPENDENT_REVIEW_AND_GO",
        "publisher": "ROOT_AUTHORITY_APPEND_ONLY_ONLY",
        "authority_scope": AUTHORIZATION_SCOPE,
    },
    "review": {
        "keys": frozenset(
            {
                "schema_version", "artifact_type", "status", "created_utc",
                "attempt_id", "amendment", "amendment_v4", "amendment_v5", "code_seal", "authorization",
                "code_identities", "output_namespace", "independent_checks",
                "authority_scope", "verdict", "publisher",
            }
        ),
        "artifact_type": "TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_REVIEW_V5",
        "status": "PASS_V5_EXECUTION_REVIEW_PENDING_INDEPENDENT_GO",
        "publisher": "INDEPENDENT_REVIEWER_APPEND_ONLY_ONLY",
        "authority_scope": DOCUMENTARY_SCOPE,
    },
    "go": {
        "keys": frozenset(
            {
                "schema_version", "artifact_type", "status", "created_utc",
                "attempt_id", "amendment", "amendment_v4", "amendment_v5", "code_seal", "authorization",
                "independent_review", "code_identities", "output_namespace",
                "required_commands", "authority_scope", "single_attempt", "publisher",
            }
        ),
        "artifact_type": "TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_GO_V5",
        "status": "GO_V5_SINGLE_TARGET_FREE_ATTEMPT",
        "publisher": "INDEPENDENT_GO_REVIEWER_APPEND_ONLY_ONLY",
        "authority_scope": GO_SCOPE,
    },
    "postrun": {
        "keys": frozenset(
            {
                "schema_version", "artifact_type", "status", "created_utc",
                "attempt_id", "amendment", "amendment_v4", "amendment_v5", "code_seal", "authorization",
                "independent_review", "independent_go", "code_identities",
                "audit_result", "staged_output_identities", "authority_scope",
                "verdict", "publisher",
            }
        ),
        "artifact_type": "TRACK_A_TARGET_FREE_DUPLICATE_POSTRUN_PASS_V5",
        "status": "PASS_V5_NO_FIT_POSTRUN_AUDIT_PENDING_FINAL_SEAL",
        "publisher": "INDEPENDENT_POSTRUN_REVIEWER_APPEND_ONLY_ONLY",
        "authority_scope": POSTRUN_SCOPE,
    },
}

REVIEW_CHECK_KEYS = frozenset(
    {
        "amendment_identity", "code_seal_identity", "authorization_identity",
        "code_identities", "test_evidence", "bound_inputs", "runtime_identity",
        "output_namespace_zero_state", "single_attempt_budget", "required_commands",
        "authority_scope", "chronology",
    }
)
SINGLE_ATTEMPT = {
    "attempt_number": 1,
    "output_root_was_absent_at_go": True,
    "retry_allowed": False,
    "prior_v5_failure_incident": None,
    "manifest_was_absent_at_go": True,
}
REQUIRED_NEXT_CONTROLS = {
    "authorization": AUTHORIZATION_RELATIVE,
    "independent_review": REVIEW_RELATIVE,
    "independent_go": GO_RELATIVE,
    "postrun_pass": POSTRUN_RELATIVE,
}
REQUIRED_REVIEW = {
    "path": REVIEW_RELATIVE,
    "status_required": "PASS_V5_EXECUTION_REVIEW_PENDING_INDEPENDENT_GO",
}
REQUIRED_GO = {
    "path": GO_RELATIVE,
    "status_required": "GO_V5_SINGLE_TARGET_FREE_ATTEMPT",
}

TEST_EVIDENCE_KEYS = frozenset(
    {
        "commands", "results", "all_exit_codes_zero",
        "all_expected_test_files_covered", "network_denied", "pycache_disabled",
        "code_identities_revalidated", "completed_utc",
    }
)
TEST_RESULT_KEYS = frozenset(
    {"test_file", "exit_code", "passed", "skipped", "deselected", "summary"}
)

PROCESS_CAPTURE_KEYS = frozenset(
    {
        "command", "exit_code", "stdout_payload", "stdout_size_bytes",
        "stdout_sha256", "stderr_size_bytes", "stderr_sha256",
        "wall_seconds_observed", "full_stdout_identity_instrumented",
    }
)
AUDIT_RESULT_KEYS = frozenset({*PROCESS_CAPTURE_KEYS, "runner_result"})
AUDITOR_STDOUT_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "status", "attempt_id", "amendment", "amendment_v4", "amendment_v5",
        "authorization", "independent_go", "code_identities",
        "staged_output_identities", "output_counts", "decision", "poststate",
        "files_written", "network_requests", "models_fit", "forbidden_reads",
    }
)
AUDITOR_FORBIDDEN_READS = {
    "labels": False,
    "arrays_2024_2025": False,
    "source_parquet_logical_values": False,
}
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

AUDITOR_OUTPUT_COUNTS = {
    "aliases": 2,
    "runner_staged_outputs": 7,
    "metrics_rows": 333,
    "stability_rows": 348,
    "oof_rows": 352_512,
    "fit_ledger_rows": 1_332,
    "distribution_records": 624,
    "endpoint_pooled_metric_records": 3,
    "derived_rows": 414_720,
}
AUDITOR_DECISION_KEYS = frozenset(
    {"status", "selected_family", "global_ambiguity"}
)
AUDITOR_POSTSTATE = {
    "output_root_entries": 3,
    "transaction_entries": 7,
    "final_publication_files": 0,
    "manifest_present": False,
    "family_lock_present": False,
    "temporary_files": 0,
    "active_locks": 0,
}
RUNNER_STDOUT_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "status", "attempt_id", "decision",
        "output_counts", "staged_output_identities",
        "threadpool_info_before_inside_after",
        "no_forbidden_access_attestation", "manifest_present",
        "family_lock_present",
    }
)
THREADPOOL_PHASES = (
    "BEFORE_EXPLICIT_CONTEXT",
    "INSIDE_CONTEXT",
    "AFTER_CONTEXT",
)
THREADPOOL_EVIDENCE_KEYS = frozenset(
    {
        "capture_phases", "capture_counts", "event_count",
        "event_chain_sha256", "first_observation_by_phase",
        "last_observation_by_phase",
    }
)
THREADPOOL_EVENT_KEYS = frozenset(
    {"event_ordinal", "phase", "label", "threadpools"}
)
THREADPOOL_OBSERVATION_KEYS = frozenset(
    {
        "name", "path", "user_api", "internal_api", "prefix", "version",
        "threading_layer", "architecture", "num_threads",
    }
)
NATIVE_LIBRARY_ORDER = (
    "numpy_openblas",
    "scipy_openblas",
    "sklearn_vcomp140",
)
THREADPOOL_BEFORE_LABEL = (
    "RUNTIME_IDENTITY_VALIDATED_BEFORE_ANY_FIT_OR_PREDICTION"
)
DECISION_FORBIDDEN_ACCESS_ATTESTATION = {
    "labels_read": False,
    "arrays_2024_2025_read": False,
    "network_requests": 0,
    "generation_model_fit_units": 0,
    "submission_csv_files_written": 0,
}

DECISION_TOP_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "status", "raw_component_order",
        "raw_component_results", "endpoint_veto_results", "global_ambiguity",
        "global_ambiguity_reasons", "family_clear_pass_results", "fixed_priority",
        "selected_family", "selected_raw_columns", "selected_derived_columns",
        "all_threshold_float_hex_witnesses", "all_tied_witnesses",
        "no_label_network_future_year_model_submission_attestation",
    }
)
RAW_DECISION_KEYS = frozenset(
    {
        "response", "classification", "counts_as_nonredundant",
        "independent_estimator", "parent_reference_estimators",
        "parent_redundancy_boolean", "affine_qualifying_predictors",
        "cell_stability_pass", "distribution_stability_pass",
        "boundary_equality_flags", "ambiguity_reason",
    }
)
ENDPOINT_DECISION_KEYS = frozenset(
    {
        "diagnostic", "verdict", "source_component_estimator_binding",
        "pooled_stable_margin_pass", "cell_stability_pass",
        "distribution_stability_pass", "boundary_equality_flags",
        "clear_pass", "ambiguity_reason",
    }
)
THRESHOLD_WITNESS_KEYS = frozenset(
    {
        "cutoff_name", "scope", "diagnostic", "value", "value_float_hex",
        "cutoff_float_hex", "boundary_equal",
    }
)
TIED_WITNESS_KEYS = frozenset(
    {
        "tie_kind", "diagnostic", "estimators", "metric_name",
        "metric_values_float_hex", "resolution", "global_ambiguity",
    }
)
ESTIMATOR_ENUM = ("RIDGE_PIPELINE", "EXTRA_TREES")
FOLD_ENUM = ("2022_H1", "2022_H2", "2023_H1", "2023_H2")
PREMODEL_CLASSIFICATIONS = (
    "CLEAR_PHYSICAL_OR_COVERAGE_FAILURE",
    "CLEAR_NONNOVEL_CONSTANT",
)
AMBIGUOUS_CLASSIFICATIONS = (
    "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE",
    "AMBIGUOUS_THRESHOLD_EQUALITY",
    "AMBIGUOUS_PARENT_ESTIMATOR_TIE",
    "AMBIGUOUS_GRAY_ZONE",
)
PASS_CLASSIFICATION = "PASS_NONREDUNDANT_MARGIN"
ENDPOINT_DIAGNOSTICS = (
    "ENDPOINT_DU_925_MINUS_1000",
    "ENDPOINT_DV_925_MINUS_1000",
    "ENDPOINT_VECTOR_SHEAR_MAG",
)
ENDPOINT_SOURCE_RESPONSES = {
    "ENDPOINT_DU_925_MINUS_1000": ("UGRD_925mb", "UGRD_1000mb"),
    "ENDPOINT_DV_925_MINUS_1000": ("VGRD_925mb", "VGRD_1000mb"),
    "ENDPOINT_VECTOR_SHEAR_MAG": (
        "UGRD_925mb", "VGRD_925mb", "UGRD_1000mb", "VGRD_1000mb",
    ),
}
ENDPOINT_VERDICTS = (
    "PASS_ENDPOINT_NONREDUNDANT_MARGIN",
    "CLEAR_ENDPOINT_VETO_FAILURE",
    "AMBIGUOUS_THRESHOLD_EQUALITY",
    "AMBIGUOUS_RAW_ESTIMATOR_TIE",
)
TIE_KINDS = ("INDEPENDENT_RMSE", "PARENT_MAX_R2")
TIE_RESOLUTIONS = (
    "GLOBAL_AMBIGUITY",
    "SHARED_PARENT_REDUNDANCY_TRUE",
    "SHARED_PARENT_REDUNDANCY_FALSE",
    "MASKED_BY_PRIOR_AFFINE_REDUNDANCY",
)
CUTOFF_VALUES = {
    "AFFINE_ABS_PEARSON_0.9999": 0.9999,
    "AFFINE_ABS_SPEARMAN_0.9999": 0.9999,
    "AFFINE_NRMSE_0.01": 0.01,
    "PARENT_R2_0.98": 0.98,
    "PARENT_NRMSE_0.10": 0.10,
    "INDEPENDENT_R2_EXACT_DUPLICATE_0.995": 0.995,
    "INDEPENDENT_NRMSE_EXACT_DUPLICATE_0.02": 0.02,
    "INDEPENDENT_R2_STABLE_MARGIN_0.9925": 0.9925,
    "INDEPENDENT_NRMSE_STABLE_MARGIN_0.025": 0.025,
    "CELL_R2_0.995": 0.995,
    "CELL_NRMSE_0.02": 0.02,
    "CELL_WEAK_FLOOR_NRMSE_0.01": 0.01,
    "MONTHLY_IQR_RATIO_0.1": 0.1,
    "MONTHLY_IQR_RATIO_10": 10.0,
    "YEAR_PSI_0.5": 0.5,
}
LOCK_TOP_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "status", "selected_family",
        "ordered_raw_columns", "ordered_derived_columns", "units", "grib_selectors",
        "site_group_scope", "spatial_transforms", "parent_and_independent_gate_results",
        "bound_input_identities", "code_config_runtime_identities",
        "postrun_pass_identity", "no_label_attestation", "downstream_authority",
    }
)
MANIFEST_TOP_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "status", "terminal_state", "created_utc",
        "code_config_runtime_identities", "control_identities", "alias_identities",
        "output_identities", "planned_completed_skipped_fit_counts", "oof_slot_counts",
        "threadpool_info_before_inside_after", "no_forbidden_access_attestation",
        "manifest_is_last_commit_marker",
    }
)
LOCK_ARTIFACT_TYPE = "TARGET_FREE_FAMILY_LOCK_V5"
MANIFEST_ARTIFACT_TYPE = "TRACK_A_TARGET_FREE_RUN_MANIFEST_V5"
MANIFEST_STATUS = "COMMITTED_V5_TARGET_FREE_DECISION"
LOCK_STATUS_BY_DECISION_STATUS = {
    "PASS_ONE_TARGET_FREE_FAMILY_SELECTED_PENDING_INDEPENDENT_POSTRUN_SEAL": (
        "LOCKED_ONE_TARGET_FREE_FAMILY_PENDING_SEPARATE_FULL_EXPANSION_AUTHORITY"
    ),
    "STOP_FAMILY_AMBIGUOUS": "AMBIGUOUS_STOP_FAMILY_AMBIGUOUS",
    "STOP_NO_TARGET_FREE_INCREMENTAL_FAMILY": (
        "NO_LOCK_STOP_NO_TARGET_FREE_INCREMENTAL_FAMILY"
    ),
}
SITE_GROUP_SCOPE_KEYS = (
    "external_site_rows",
    "external_group_rows",
    "valid_timestamps",
    "canonical_site_order",
    "site_join_key_exact",
    "group_join_key_exact",
    "metadata_parity_required",
)
SPATIAL_TRANSFORM_KEYS = ("spatial_method", "group_construction")
CODE_CONFIG_RUNTIME_KEYS = (
    "amendment",
    "amendment_v4",
    "amendment_v5",
    "code_identities",
    "runtime_identity",
)
MANIFEST_CONTROL_IDENTITY_KEYS = (
    "code_seal",
    "authorization",
    "independent_review",
    "independent_go",
    "postrun_pass",
)
MANIFEST_FIT_COUNT_KEYS = (
    "planned_counts",
    "executed_counts",
    "internal_method_call_counts",
    "skipped_counts",
    "zero_fit_counts",
)
MANIFEST_OOF_COUNT_KEYS = (
    "planned_oof_slots",
    "completed_oof_slots",
    "skipped_oof_slots",
    "independent_selected_oof_slots",
)
MANIFEST_OUTPUT_ORDER = (*RUNNER_PUBLICATION_ORDER, "TARGET_FREE_FAMILY_LOCK.json")
GRIB_SELECTOR_KEYS = ("variable", "level")
FIT_LEDGER_TOP_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "status", "planned_counts",
        "executed_counts", "internal_method_call_counts", "skipped_counts",
        "zero_fit_counts", "unit_slots",
    }
)
FIT_LEDGER_PLANNED_COUNTS = {
    "primary_model_units": 72,
    "analytic_affine_units": 1_260,
    "total_decision_units": 1_332,
    "fit_ledger_rows": 1_332,
}
FIT_LEDGER_EXECUTED_KEYS = frozenset(
    {
        "completed_primary_model_units",
        "completed_analytic_affine_units",
        "completed_total_decision_units",
    }
)
FIT_LEDGER_INTERNAL_METHOD_KEYS = frozenset(
    {
        "sklearn.pipeline.Pipeline.fit",
        "sklearn.preprocessing.StandardScaler.fit",
        "sklearn.linear_model.Ridge.fit",
        "sklearn.ensemble.ExtraTreesRegressor.fit",
        "total_method_calls",
    }
)
FIT_LEDGER_SKIPPED_KEYS = frozenset(
    {
        "skipped_primary_model_units",
        "skipped_analytic_affine_units",
        "skipped_total_decision_units",
    }
)
FIT_LEDGER_ZERO_COUNTS = {
    "derived": 0,
    "group": 0,
    "hyperparameter_or_seed_search": 0,
}
DISTRIBUTION_TOP_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "status", "scope_order",
        "diagnostic_order", "endpoint_pooled_metric_records", "records",
    }
)
CENSUS_ALIAS_TOP_KEYS = frozenset(
    {
        "access_ledger", "artifact_type", "created_utc", "day_summary",
        "field_range_census", "field_summary", "labels_read", "models_fit",
        "object_census", "parent_preregister_manifest", "progress_checkpoints",
        "raw_download_plan", "raw_downloaded_bytes", "raw_network_requests",
        "reproduction_code", "schema_version", "submission_csv_created",
        "summary", "test_code",
    }
)
CENSUS_ALIAS_IDENTITY_FIELDS = (
    "access_ledger",
    "day_summary",
    "field_range_census",
    "field_summary",
    "object_census",
    "parent_preregister_manifest",
    "raw_download_plan",
    "reproduction_code",
    "summary",
    "test_code",
)

RFC3339_MICROSECOND_Z = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)
LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ATTEMPT_ID = re.compile(
    r"^target_free_duplicate_v5__[0-9]{8}T[0-9]{12}Z$"
)
PYTEST_SUMMARY = re.compile(
    r"^(?P<passed>[0-9]+) passed"
    r"(?:, (?P<skipped>[0-9]+) skipped)?"
    r"(?:, (?P<deselected>[0-9]+) deselected)? "
    r"in (?P<duration_whole>[0-9]+)(?:\.(?P<duration_fraction>[0-9]+))?s"
    r"(?: \((?P<hours>0|[1-9][0-9]*):(?P<minutes>[0-5][0-9]):"
    r"(?P<seconds>[0-5][0-9])\))?$"
)
NONFINITE_TEXT = frozenset(
    {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
)


class TargetFreeSealError(RuntimeError):
    """A fail-closed V5 sealer contract check failed."""


class DynamicIdentityPending(TargetFreeSealError):
    """The six code/test files are not all present for the final hard bind."""


class PostPublicationError(TargetFreeSealError):
    """A final name exists but a subsequent verification step failed."""

    def __init__(self, message: str, identity_record: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.identity_record = dict(identity_record)


class PartialPublicationError(TargetFreeSealError):
    """A V5 publication became partial and therefore requires V6 supersession."""

    def __init__(self, message: str, published: Sequence[str]) -> None:
        super().__init__(message)
        self.published = tuple(published)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TargetFreeSealError(message)


def _exact_tree_equal(observed: Any, expected: Any) -> bool:
    """Compare frozen JSON-like values without bool/int coercion."""

    if isinstance(expected, Mapping):
        return (
            isinstance(observed, Mapping)
            and set(observed) == set(expected)
            and all(
                _exact_tree_equal(observed[key], expected[key])
                for key in expected
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(expected)
            and all(
                _exact_tree_equal(left, right)
                for left, right in zip(observed, expected)
            )
        )
    return type(observed) is type(expected) and observed == expected


def _require_exact_tree(observed: Any, expected: Any, *, label: str) -> None:
    _require(_exact_tree_equal(observed, expected), f"{label} mismatch")


def _lexists(path: Path) -> bool:
    return os.path.lexists(str(path))


def _linklike(path: Path) -> bool:
    return path.is_symlink() or (
        hasattr(path, "is_junction") and path.is_junction()
    )


def _reject_nonfinite_constant(value: str) -> Any:
    raise TargetFreeSealError(f"non-finite JSON literal is forbidden: {value}")


def _require_finite_json(value: Any, *, label: str, pointer: str = "$") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} contains non-finite number at {pointer}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite_json(child, label=label, pointer=f"{pointer}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_json(child, label=label, pointer=f"{pointer}[{index}]")


def strict_json_loads(raw: str, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TargetFreeSealError(f"{label} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=_reject_nonfinite_constant,
        )
    except TargetFreeSealError:
        raise
    except Exception as exc:
        raise TargetFreeSealError(f"malformed {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} root must be an object")
    _require_finite_json(payload, label=label)
    return payload


def strict_json_load(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not _linklike(path), f"{label} is absent or link-like")
    raw = path.read_bytes()
    _require(not raw.startswith(b"\xef\xbb\xbf"), f"{label} has a UTF-8 BOM")
    _require(b"\r" not in raw and b"\x00" not in raw, f"{label} has forbidden CR/NUL bytes")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TargetFreeSealError(f"{label} is not strict UTF-8") from exc
    payload = strict_json_loads(text, label=label)
    _require(
        pretty_json_bytes(payload) == raw,
        f"{label} deterministic JSON serialization mismatch",
    )
    return payload


def pretty_json_bytes(payload: Any) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    except (TypeError, ValueError) as exc:
        raise TargetFreeSealError(f"JSON payload is not finite/serializable: {exc}") from exc
    return (text + "\n").encode("utf-8")


def compact_json_bytes(payload: Any, *, ensure_ascii: bool = True) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=ensure_ascii,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TargetFreeSealError(f"canonical JSON is not finite/serializable: {exc}") from exc
    return text.encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_no_link_chain(path: Path, boundary: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(path))
    root = Path(os.path.abspath(boundary))
    _require(lexical == root or root in lexical.parents, f"{label} escapes its boundary")
    current = lexical
    while True:
        _require(not _linklike(current), f"{label} contains symlink/junction: {current}")
        if current == root:
            break
        current = current.parent
    return lexical


def path_under(root: Path, relative: str, *, label: str) -> Path:
    _require(
        isinstance(relative, str)
        and bool(relative)
        and "\\" not in relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        f"{label} is not a canonical forward-slash relative path",
    )
    return _require_no_link_chain(root / Path(relative), root, label=label)


def normalized_absolute(path: Path) -> str:
    return str(Path(os.path.abspath(path))).replace("\\", "/")


def lexical_absolute(path: Path) -> Path:
    """Make a path absolute without following a caller-supplied link."""

    return Path(os.path.abspath(path))


def file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    _require(path.is_file() and not _linklike(path), f"identity file absent/link-like: {path}")
    record_path = (
        path.relative_to(relative_to).as_posix()
        if relative_to is not None
        else normalized_absolute(path)
    )
    return {
        "path": record_path,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_identity_record(
    record: Any,
    expected_path: Path,
    *,
    label: str,
    relative_to: Path | None,
    rehash: bool = True,
) -> dict[str, Any]:
    _require(isinstance(record, Mapping), f"{label} identity is not an object")
    _require(set(record) == FILE_IDENTITY_KEYS, f"{label} identity schema mismatch")
    expected_text = (
        expected_path.relative_to(relative_to).as_posix()
        if relative_to is not None
        else normalized_absolute(expected_path)
    )
    _require(record.get("path") == expected_text, f"{label} identity path mismatch")
    _require(
        isinstance(record.get("size_bytes"), int)
        and not isinstance(record["size_bytes"], bool)
        and int(record["size_bytes"]) >= 0,
        f"{label} identity size is invalid",
    )
    _require(
        isinstance(record.get("sha256"), str)
        and LOWER_SHA256.fullmatch(str(record["sha256"])) is not None,
        f"{label} identity SHA-256 is invalid",
    )
    if rehash:
        observed = file_identity(expected_path, relative_to=relative_to)
        _require(dict(record) == observed, f"{label} physical identity mismatch")
    return dict(record)


def amendment_identity() -> dict[str, Any]:
    return {
        "path": AMENDMENT_RELATIVE,
        "size_bytes": AMENDMENT_SIZE_BYTES,
        "sha256": AMENDMENT_SHA256,
        "canonical_size_bytes": AMENDMENT_CANONICAL_SIZE_BYTES,
        "canonical_sha256": AMENDMENT_CANONICAL_SHA256,
    }


def amendment_v4_identity() -> dict[str, Any]:
    return {
        "path": AMENDMENT_V4_RELATIVE,
        "size_bytes": AMENDMENT_V4_SIZE_BYTES,
        "sha256": AMENDMENT_V4_SHA256,
        "canonical_size_bytes": AMENDMENT_V4_CANONICAL_SIZE_BYTES,
        "canonical_sha256": AMENDMENT_V4_CANONICAL_SHA256,
    }


def incident_identity() -> dict[str, Any]:
    return {
        "path": INCIDENT_RELATIVE,
        "size_bytes": INCIDENT_SIZE_BYTES,
        "sha256": INCIDENT_SHA256,
        "canonical_size_bytes": INCIDENT_CANONICAL_SIZE_BYTES,
        "canonical_sha256": INCIDENT_CANONICAL_SHA256,
    }


def amendment_v5_identity() -> dict[str, Any]:
    return {
        "path": AMENDMENT_V5_RELATIVE,
        "size_bytes": AMENDMENT_V5_SIZE_BYTES,
        "sha256": AMENDMENT_V5_SHA256,
        "canonical_size_bytes": AMENDMENT_V5_CANONICAL_SIZE_BYTES,
        "canonical_sha256": AMENDMENT_V5_CANONICAL_SHA256,
    }


def require_created_utc(value: Any, *, label: str) -> str:
    _require(isinstance(value, str), f"{label} created_utc is absent")
    _require(RFC3339_MICROSECOND_Z.fullmatch(value) is not None, f"{label} timestamp format mismatch")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise TargetFreeSealError(f"{label} timestamp is invalid") from exc
    _require(parsed.microsecond == int(value[-7:-1]), f"{label} timestamp lost precision")
    return value


def utc_now_exact() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def require_attempt_id(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str) and ATTEMPT_ID.fullmatch(value) is not None,
        f"{label} attempt_id format mismatch",
    )
    timestamp = value.removeprefix("target_free_duplicate_v5__")
    try:
        parsed = datetime.strptime(timestamp, "%Y%m%dT%H%M%S%fZ")
    except ValueError as exc:
        raise TargetFreeSealError(f"{label} attempt_id calendar timestamp invalid") from exc
    _require(
        parsed.strftime("%Y%m%dT%H%M%S%fZ") == timestamp,
        f"{label} attempt_id timestamp did not round-trip exactly",
    )
    return value


def _validate_v3_amendment(artifact_root: Path) -> dict[str, Any]:
    artifact_root = Path(os.path.abspath(artifact_root))
    path = path_under(artifact_root, AMENDMENT_RELATIVE, label="V3 amendment")
    observed = file_identity(path, relative_to=artifact_root)
    _require(observed == {k: v for k, v in amendment_identity().items() if k in FILE_IDENTITY_KEYS}, "V3 amendment physical identity mismatch")
    payload = strict_json_load(path, label="V3 amendment")
    canonical = compact_json_bytes(payload, ensure_ascii=False)
    _require(len(canonical) == AMENDMENT_CANONICAL_SIZE_BYTES, "V3 amendment canonical size mismatch")
    _require(hashlib.sha256(canonical).hexdigest() == AMENDMENT_CANONICAL_SHA256, "V3 amendment canonical SHA mismatch")
    strict = payload.get("strict_schema")
    _require(isinstance(strict, Mapping), "V3 strict_schema is absent")
    _require(set(payload) == set(strict.get("top_level_exact_keys", ())), "V3 top-level schema mismatch")
    _require(len(payload) == 26 == strict.get("top_level_exact_key_count"), "V3 top-level key count mismatch")
    counts = strict.get("top_level_object_key_counts")
    _require(isinstance(counts, Mapping), "V3 nested key-count contract absent")
    for name, count in counts.items():
        _require(isinstance(payload.get(name), Mapping), f"V3 nested object absent: {name}")
        _require(len(payload[name]) == count, f"V3 nested object key count mismatch: {name}")
    require_created_utc(payload.get("created_utc"), label="V3 amendment")
    _require(payload.get("schema_version") == 3, "V3 schema version mismatch")
    _require(
        payload.get("status")
        == "FROZEN_FOR_CODE_AND_TEST_IMPLEMENTATION_ONLY_PENDING_SEPARATE_CODE_SEAL_REVIEW_AND_GO_BEFORE_VALUE_READ_OR_FIT",
        "V3 status mismatch",
    )
    authority = payload.get("authority_scope")
    _require(isinstance(authority, Mapping), "V3 authority_scope absent")
    for key in (
        "decoded_or_predictor_parquet_value_read_authorized", "sklearn_fit_authorized",
        "analytic_affine_fit_authorized", "execution_output_publication_authorized",
        "network_authorized", "label_or_scada_read_authorized", "gpu_authorized",
        "submission_csv_authorized",
    ):
        _require(authority.get(key) is False, f"V3 documentary authority escalated: {key}")
    _require(authority.get("code_and_test_implementation_authorized") is True, "V3 code implementation authority absent")
    _require(authority.get("authorized_code_and_test_paths") == list(V3_CODE_ROLE_PATHS.values()), "V3 code/test path order mismatch")

    output = payload.get("output_contract")
    _require(isinstance(output, Mapping), "V3 output contract absent")
    _require(output.get("output_root") == V3_OUTPUT_ROOT_RELATIVE, "V3 output root mismatch")
    _require(output.get("positive_complete_paths_exact") == [f"{V3_OUTPUT_ROOT_RELATIVE}{name}" for name in V3_EXACT11_SCHEMA_ORDER], "V3 positive exact11 mismatch")
    _require(output.get("completed_negative_complete_paths_exact") == [f"{V3_OUTPUT_ROOT_RELATIVE}{name}" for name in V3_EXACT11_SCHEMA_ORDER], "V3 negative exact11 mismatch")
    _require(output.get("final_publication_order_positive_exact") == list(V3_FINAL_PUBLICATION_ORDER), "V3 positive publication order mismatch")
    _require(output.get("final_publication_order_negative_exact") == list(V3_FINAL_PUBLICATION_ORDER), "V3 negative publication order mismatch")
    _require(output.get("runner_staged_paths_exact") == V3_RUNNER_STAGED_RELATIVES, "V3 runner staged paths mismatch")
    _require(output.get("sealer_staged_paths_exact") == V3_SEALER_STAGED_RELATIVES, "V3 sealer staged paths mismatch")
    _require(output.get("single_v3_attempt_only") is True, "V3 single-attempt flag mismatch")
    _require(output.get("attempt_numbered_subdirectories_allowed") is False, "V3 attempt subdirectories unexpectedly allowed")
    _require(output.get("v3_root_reuse_after_any_failure_or_partial_publication_allowed") is False, "V3 root reuse unexpectedly allowed")
    _require(output.get("no_overwrite") is True, "V3 no-overwrite flag mismatch")
    _require(
        output.get("manifest_publish_order")
        == "LAST_COMMIT_MARKER_AFTER_EVERY_REQUIRED_FINAL_FOR_THAT_TERMINAL_STATE_IS_REHASHED",
        "V3 manifest-last contract mismatch",
    )
    _require(output.get("persistent_hardlink_to_source_forbidden") is True, "V3 persistent source hardlink unexpectedly allowed")
    aliases = output.get("upstream_alias_closure", {}).get("aliases")
    _require(
        isinstance(aliases, Mapping)
        and len(aliases) == len(ALIAS_ORDER)
        and set(aliases) == set(ALIAS_ORDER),
        "V3 alias set mismatch",
    )
    for name, frozen in V3_ALIAS_SPECS.items():
        record = aliases.get(name)
        _require(isinstance(record, Mapping), f"V3 alias absent: {name}")
        _require(record.get("exact_byte_source") == frozen["source"], f"V3 alias source mismatch: {name}")
        _require(record.get("final_path") == frozen["final"], f"V3 alias final mismatch: {name}")
        _require(record.get("size_bytes") == frozen["size_bytes"], f"V3 alias size mismatch: {name}")
        _require(record.get("sha256") == frozen["sha256"], f"V3 alias SHA mismatch: {name}")
        _require(tuple(record.get("semantic_authority_bundle", ())) == frozen["bundle"], f"V3 alias bundle mismatch: {name}")

    gate = payload.get("implementation_release_gate")
    _require(isinstance(gate, Mapping), "V3 implementation gate absent")
    _require(gate.get("future_code_seal_path") == "prereg/target_free_duplicate_code_seal_v3.json", "V3 code-seal path mismatch")
    _require(gate.get("future_execution_authorization_path") == "prereg/target_free_duplicate_execution_authorization_v3.json", "V3 authorization path mismatch")
    _require(gate.get("future_independent_review_path") == "independent_redteam/TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_REVIEW_V3.json", "V3 review path mismatch")
    _require(gate.get("future_independent_go_path") == "independent_redteam/TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_GO_V3.json", "V3 GO path mismatch")
    _require(gate.get("future_postrun_independent_pass_path") == "independent_redteam/TRACK_A_TARGET_FREE_DUPLICATE_POSTRUN_PASS_V3.json", "V3 postrun path mismatch")
    _require(gate.get("required_sealer") == V3_CODE_ROLE_PATHS["sealer"], "V3 sealer path mismatch")
    _require(gate.get("required_sealer_test") == V3_CODE_ROLE_PATHS["sealer_test"], "V3 sealer-test path mismatch")
    _require(gate.get("execution_authorized_now") is False and gate.get("value_read_authorized_now") is False and gate.get("fit_authorized_now") is False, "V3 implementation gate grants execution")
    return payload


def _validate_v4_pyarrow_signature(payload: Mapping[str, Any]) -> None:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    evidence = payload["pyarrow_22_0_0_evidence"]
    _require(
        pa.__version__ == evidence["runtime_version_exact"] == "22.0.0",
        "V4 PyArrow runtime version mismatch",
    )
    signature = inspect.signature(pq.write_table)
    explicit = set(V4_EXPLICIT_WRITE_TABLE_KWARGS)
    remaining = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name not in explicit | {"table", "where", "kwargs"}
        and parameter.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }
    _require_exact_tree(
        remaining,
        V4_UNLISTED_WRITE_TABLE_DEFAULTS,
        label="V4 PyArrow unlisted write_table runtime defaults",
    )
    _require(
        "kwargs" in signature.parameters
        and signature.parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD,
        "V4 PyArrow write_table variadic keyword boundary mismatch",
    )


def validate_amendment_v4(
    artifact_root: Path, v3: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(os.path.abspath(artifact_root))
    path = path_under(root, AMENDMENT_V4_RELATIVE, label="V4 amendment")
    observed = file_identity(path, relative_to=root)
    _require_exact_tree(
        observed,
        {key: value for key, value in amendment_v4_identity().items() if key in FILE_IDENTITY_KEYS},
        label="V4 amendment physical identity",
    )
    payload = strict_json_load(path, label="V4 amendment")
    canonical = compact_json_bytes(payload, ensure_ascii=False)
    _require(
        len(canonical) == AMENDMENT_V4_CANONICAL_SIZE_BYTES
        and hashlib.sha256(canonical).hexdigest() == AMENDMENT_V4_CANONICAL_SHA256,
        "V4 amendment canonical identity mismatch",
    )
    strict = payload.get("strict_schema")
    _require(
        isinstance(strict, Mapping)
        and set(payload) == AMENDMENT_V4_TOP_KEYS
        and set(payload) == set(strict.get("top_level_exact_keys", ()))
        and len(payload) == strict.get("top_level_exact_key_count") == 18,
        "V4 amendment top schema mismatch",
    )
    counts = strict.get("top_level_object_key_counts")
    _require(
        isinstance(counts, Mapping)
        and set(counts)
        == {key for key, value in payload.items() if isinstance(value, Mapping)},
        "V4 amendment nested count map mismatch",
    )
    for name, count in counts.items():
        _require(
            type(count) is int
            and isinstance(payload[name], Mapping)
            and len(payload[name]) == count,
            f"V4 amendment nested key count mismatch: {name}",
        )
    _require(
        payload.get("schema_version") == 4
        and type(payload["schema_version"]) is int
        and payload.get("artifact_type")
        == "TARGET_FREE_MULTISEASON_PREREGISTRATION_APPEND_ONLY_DOCUMENTARY_CORRECTION_V4"
        and payload.get("status")
        == "PASS_DOCUMENTARY_V4_PARQUET_API_AND_CSV_BOUNDARY_SERIALIZATION_CORRECTION_ONLY_NO_EXECUTION_AUTHORITY"
        and payload.get("canonical_path") == AMENDMENT_V4_RELATIVE
        and payload.get("created_utc") == AMENDMENT_V4_CREATED_UTC,
        "V4 amendment envelope mismatch",
    )
    require_created_utc(payload["created_utc"], label="V4 amendment")
    _require_exact_tree(
        payload.get("bound_v3_amendment"), amendment_identity(),
        label="V4 bound V3 amendment",
    )
    _require(
        _utc_order_value(v3["created_utc"], label="V3 amendment")
        < _utc_order_value(payload["created_utc"], label="V4 amendment"),
        "V3/V4 amendment chronology mismatch",
    )

    scope = payload.get("correction_scope")
    _require(isinstance(scope, Mapping) and len(scope) == 6, "V4 correction scope schema mismatch")
    _require_exact_tree(
        scope.get("superseded_pointers_exact_order"), list(V4_CORRECTION_PATHS),
        label="V4 correction pointer order",
    )
    _require(
        scope.get("correction_count_exact") == 2
        and type(scope["correction_count_exact"]) is int
        and scope.get("all_other_v3_parsed_semantic_pointers_unchanged") is True
        and scope.get("scope_expansion_forbidden") is True,
        "V4 exact-two correction scope mismatch",
    )
    old_literals = {
        V4_CORRECTION_PATHS[0]: v3["output_contract"]["output_serialization_exact"]["parquet"],
        V4_CORRECTION_PATHS[1]: v3["output_contract"]["output_serialization_exact"]["csv_boundary_flags"],
    }
    replacements = {
        V4_CORRECTION_PATHS[0]: V4_PARQUET_LITERAL,
        V4_CORRECTION_PATHS[1]: V4_CSV_BOUNDARY_LITERAL,
    }
    _require_exact_tree(scope.get("v3_literals_exact"), old_literals, label="V4 old literal binding")
    _require_exact_tree(
        scope.get("replacement_literals_exact"), replacements,
        label="V4 replacement literal binding",
    )

    parquet = payload.get("corrected_parquet_serialization")
    _require(isinstance(parquet, Mapping) and len(parquet) == 4, "V4 Parquet correction schema mismatch")
    _require(
        parquet.get("corrected_v3_literal") == V4_PARQUET_LITERAL,
        "V4 corrected Parquet literal mismatch",
    )
    _require_exact_tree(
        parquet.get("explicit_write_table_kwargs_exact"),
        V4_EXPLICIT_WRITE_TABLE_KWARGS,
        label="V4 explicit write_table kwargs",
    )
    _require(
        len(parquet.get("native_ns_validation_exact", {})) == 8
        and len(parquet.get("unchanged_non_kwarg_invariants_exact", {})) == 7
        and all(parquet["native_ns_validation_exact"].values())
        and all(
            value is True
            for key, value in parquet["unchanged_non_kwarg_invariants_exact"].items()
            if key not in {"float_columns_arrow_type_exact", "pyarrow_runtime_version_exact"}
        )
        and parquet["unchanged_non_kwarg_invariants_exact"]["float_columns_arrow_type_exact"] == "float64"
        and parquet["unchanged_non_kwarg_invariants_exact"]["pyarrow_runtime_version_exact"] == "22.0.0",
        "V4 native-ns/non-kwarg invariant mismatch",
    )
    native = parquet["native_ns_validation_exact"]
    _require(
        native["valid_time_utc_arrow_type_exact"] == "timestamp[ns, tz=UTC]"
        and native["target_operating_day_kst_arrow_type_exact"] == "date32[day]"
        and native["nanosecond_residue_witness_exact"] == 123
        and type(native["nanosecond_residue_witness_exact"]) is int,
        "V4 native-ns schema/witness mismatch",
    )

    csv_contract = payload.get("corrected_csv_boundary_serialization")
    _require(isinstance(csv_contract, Mapping) and len(csv_contract) == 6, "V4 CSV correction schema mismatch")
    _require(
        csv_contract.get("corrected_v3_literal") == V4_CSV_BOUNDARY_LITERAL
        and csv_contract.get("csv_boolean_cells_exact_values") == ["TRUE", "FALSE"]
        and csv_contract.get("boundary_equality_flags_none_representation") == "EMPTY_FIELD"
        and csv_contract.get("boundary_equality_flags_present_representation")
        == "SEMICOLON_JOINED_ACTIVE_CONTINUOUS_CUTOFF_NAMES"
        and csv_contract.get("boundary_equality_flag_order_source")
        == "component_decision_truth_table.continuous_cutoffs_exact"
        and csv_contract.get("duplicate_or_inactive_cutoff_names_forbidden") is True,
        "V4 CSV boundary serialization mismatch",
    )
    expected_temporal = {
        "valid_time_utc": {"arrow_type": "timestamp[ns, tz=UTC]", "nullable": False},
        "target_operating_day_kst": {"arrow_type": "date32[day]", "nullable": False},
    }
    _require_exact_tree(
        payload.get("affected_output_schemas_exact"),
        {
            "TARGET_FREE_DUPLICATE_OOF_V3.parquet": expected_temporal,
            "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3.parquet": expected_temporal,
        },
        label="V4 affected temporal output schemas",
    )
    authority = payload.get("authority_scope")
    _require(
        isinstance(authority, Mapping)
        and len(authority) == 16
        and authority.get("authorized_code_and_test_paths") == list(V3_CODE_ROLE_PATHS.values())
        and authority.get("code_and_test_implementation_authorized") is True
        and authority.get("documentary_correction") is True
        and all(
            value is False
            for key, value in authority.items()
            if key not in {
                "authorized_code_and_test_paths", "code_and_test_implementation_authorized",
                "documentary_correction",
            }
        ),
        "V4 documentary authority mismatch",
    )
    controls = payload.get("required_future_control_bindings")
    _require(
        isinstance(controls, Mapping)
        and len(controls) == 16
        and controls.get("code_seal_top_level_key_count") == 12
        and controls.get("execution_authorization_top_level_key_count") == 16
        and controls.get("independent_review_top_level_key_count") == 15
        and controls.get("independent_go_top_level_key_count") == 16
        and controls.get("independent_postrun_top_level_key_count") == 17
        and controls.get("auditor_stdout_top_level_key_count") == 17
        and controls.get("runner_stdout_top_level_key_count") == 11
        and controls.get("v3_identity_field_name") == "amendment"
        and controls.get("v4_identity_field_name") == "amendment_v4"
        and controls.get("runner_stdout_amendment_v4_field_forbidden") is True
        and controls.get("control_top_keyset_change_is_only_addition_of_amendment_v4") is True,
        "V4 future control binding mismatch",
    )
    final_binding = payload.get("final_lock_manifest_binding_supersession")
    _require(
        isinstance(final_binding, Mapping)
        and len(final_binding) == 7
        and final_binding.get("prior_exact_key_count") == 3
        and final_binding.get("superseding_exact_key_count") == 4
        and final_binding.get("superseding_exact_keys")
        == ["amendment", "amendment_v4", "code_identities", "runtime_identity"]
        and final_binding.get("required_in_both_final_lock_and_manifest") is True,
        "V4 final lock/manifest binding mismatch",
    )
    preserved = payload.get("preserved_v3_semantics")
    _require(
        isinstance(preserved, Mapping)
        and len(preserved) == 14
        and preserved.get("parsed_semantic_pointer_diff_count_exact") == 2
        and preserved.get("parsed_semantic_pointer_diff_paths_exact") == list(V4_CORRECTION_PATHS)
        and all(
            value is True
            for key, value in preserved.items()
            if key not in {
                "parsed_semantic_pointer_diff_count_exact",
                "parsed_semantic_pointer_diff_paths_exact",
            }
        ),
        "V4 preserved V3 semantics mismatch",
    )
    _require(
        isinstance(payload.get("prohibitions"), Mapping)
        and len(payload["prohibitions"]) == 23
        and all(value is True for value in payload["prohibitions"].values()),
        "V4 prohibition contract mismatch",
    )
    _validate_v4_pyarrow_signature(payload)
    return dict(payload)


def validate_v3_failure_incident(artifact_root: Path) -> dict[str, Any]:
    """Bind the immutable V3 terminal incident without extending its claims."""

    root = Path(os.path.abspath(artifact_root))
    path = path_under(root, INCIDENT_RELATIVE, label="bound V3 failure incident")
    observed = file_identity(path, relative_to=root)
    _require_exact_tree(
        observed,
        {key: value for key, value in incident_identity().items() if key in FILE_IDENTITY_KEYS},
        label="V3 failure incident physical identity",
    )
    payload = strict_json_load(path, label="bound V3 failure incident")
    canonical = compact_json_bytes(payload, ensure_ascii=False)
    _require(
        len(canonical) == INCIDENT_CANONICAL_SIZE_BYTES
        and hashlib.sha256(canonical).hexdigest() == INCIDENT_CANONICAL_SHA256,
        "V3 failure incident canonical identity mismatch",
    )
    strict = payload.get("strict_schema")
    _require(
        isinstance(strict, Mapping)
        and set(payload) == set(strict.get("top_level_keys_exact", ()))
        and len(payload) == strict.get("top_level_key_count") == 20,
        "V3 failure incident top schema mismatch",
    )
    counts = strict.get("nested_object_key_counts_exact")
    _require(
        isinstance(counts, Mapping)
        and counts.get("/runner_execution_capture") == 20
        and counts.get("/failure_boundary") == 22
        and counts.get("/postfailure_state") == 16
        and counts.get("/forbidden_access_attestation") == 19
        and counts.get("/required_v5_supersession") == 20,
        "V3 failure incident nested count contract mismatch",
    )
    _require(
        payload.get("schema_version") == 1
        and type(payload["schema_version"]) is int
        and payload.get("artifact_type")
        == "TARGET_FREE_DUPLICATE_V3_BOUND_INPUT_FOOTER_SCHEMA_FALSE_REJECT_INCIDENT_V1"
        and payload.get("status")
        == "SEALED_V3_BOUND_INPUT_FOOTER_SCHEMA_FALSE_REJECT_AFTER_ALIAS2_NO_VALUE_READ_NO_FIT_V5_REQUIRED"
        and payload.get("canonical_path") == INCIDENT_RELATIVE
        and payload.get("created_utc") == INCIDENT_CREATED_UTC,
        "V3 failure incident envelope mismatch",
    )
    require_created_utc(payload["created_utc"], label="V3 failure incident")
    _require_exact_tree(
        payload.get("amendments"),
        {"amendment": amendment_identity(), "amendment_v4": amendment_v4_identity()},
        label="V3 failure incident amendment lineage",
    )
    historical_controls = payload.get("control_identities")
    _require(
        isinstance(historical_controls, Mapping)
        and set(historical_controls)
        == {"code_seal", "authorization", "independent_review", "independent_go"},
        "V3 failure incident control identity set mismatch",
    )
    for role, record in historical_controls.items():
        _require(isinstance(record, Mapping), f"V3 historical control absent: {role}")
        control_path = path_under(root, str(record.get("path")), label=f"V3 historical control {role}")
        validate_identity_record(
            record,
            control_path,
            label=f"V3 historical control {role}",
            relative_to=root,
            rehash=True,
        )
        strict_json_load(control_path, label=f"V3 historical control {role}")
    historical_code = payload.get("code_identities")
    _require(
        isinstance(historical_code, Mapping)
        and set(historical_code) == set(V3_CODE_ROLE_PATHS),
        "V3 failure incident code identity set mismatch",
    )
    for role, relative in V3_CODE_ROLE_PATHS.items():
        validate_identity_record(
            historical_code[role],
            path_under(REPO, relative, label=f"V3 historical code {role}"),
            label=f"V3 historical code {role}",
            relative_to=None,
            rehash=True,
        )
    capture = payload.get("runner_execution_capture")
    _require(
        isinstance(capture, Mapping)
        and capture.get("exit_code") == 1
        and type(capture["exit_code"]) is int
        and capture.get("start_utc") is None
        and capture.get("end_utc") is None
        and capture.get("start_end_utc_instrumented") is False
        and capture.get("stdout_size_bytes") == 0
        and capture.get("stdout_sha256") == EMPTY_SHA256
        and capture.get("stderr_size_bytes") == 1_806
        and capture.get("stderr_sha256")
        == "1226af3df47347e183aa7ed047efa7cbd5e548dbc7b8e652e98100200d614332",
        "V3 failure incident process capture mismatch",
    )
    failure = payload.get("failure_boundary")
    _require(
        isinstance(failure, Mapping)
        and failure.get("arrow_handoff_authorized") is True
        and failure.get("guarded_read_table_reached") is False
        and failure.get("any_bound_parquet_logical_value_read") is False
        and failure.get("run_prepared_execution_reached") is False
        and failure.get("any_fit_reached") is False,
        "V3 failure incident boundary mismatch",
    )
    forbidden = payload.get("forbidden_access_attestation")
    _require(
        isinstance(forbidden, Mapping)
        and forbidden.get("network_requests") is None
        and forbidden.get("network_counter_persisted") is False
        and forbidden.get("bound_parquet_logical_value_reads") == 0
        and forbidden.get("decoded_predictor_value_reads") == 0
        and forbidden.get("models_fit_total") == 0
        and forbidden.get("aliases_published") == 2,
        "V3 failure incident scoped forbidden-access facts mismatch",
    )
    disposition = payload.get("v3_disposition")
    _require(
        isinstance(disposition, Mapping)
        and disposition.get("attempt_consumed") is True
        and disposition.get("retry_allowed") is False
        and disposition.get("v3_output_root_reuse_allowed") is False
        and disposition.get("partial_namespace_is_immutable") is True,
        "V3 failure incident disposition mismatch",
    )
    return dict(payload)


def validate_amendment_v5(
    artifact_root: Path,
    v3: Mapping[str, Any],
    v4: Mapping[str, Any],
    incident: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the standalone V5 documentary amendment without footer opens."""

    del incident  # its exact physical identity was validated above
    root = Path(os.path.abspath(artifact_root))
    path = path_under(root, AMENDMENT_V5_RELATIVE, label="V5 amendment")
    observed = file_identity(path, relative_to=root)
    _require_exact_tree(
        observed,
        {key: value for key, value in amendment_v5_identity().items() if key in FILE_IDENTITY_KEYS},
        label="V5 amendment physical identity",
    )
    payload = strict_json_load(path, label="V5 amendment")
    canonical = compact_json_bytes(payload, ensure_ascii=False)
    _require(
        len(canonical) == AMENDMENT_V5_CANONICAL_SIZE_BYTES
        and hashlib.sha256(canonical).hexdigest() == AMENDMENT_V5_CANONICAL_SHA256,
        "V5 amendment canonical identity mismatch",
    )
    strict = payload.get("strict_schema")
    _require(
        isinstance(strict, Mapping)
        and set(payload) == set(strict.get("top_level_exact_keys", ()))
        and len(payload) == strict.get("top_level_exact_key_count") == 24,
        "V5 amendment top schema mismatch",
    )
    counts = strict.get("top_level_object_key_counts")
    _require(isinstance(counts, Mapping), "V5 amendment nested count map absent")
    for name, count in counts.items():
        _require(
            type(count) is int
            and isinstance(payload.get(name), Mapping)
            and len(payload[name]) == count,
            f"V5 amendment nested key count mismatch: {name}",
        )
    _require(
        payload.get("schema_version") == 5
        and type(payload["schema_version"]) is int
        and payload.get("artifact_type")
        == "TARGET_FREE_MULTISEASON_PREREGISTRATION_APPEND_ONLY_EXECUTION_SUPERSESSION_V5"
        and payload.get("status")
        == "FROZEN_FOR_V5_CODE_AND_TEST_IMPLEMENTATION_ONLY_PENDING_SEPARATE_CODE_SEAL_REVIEW_AND_GO_BEFORE_FOOTER_GATE_VALUE_READ_OR_FIT"
        and payload.get("canonical_path") == AMENDMENT_V5_RELATIVE
        and payload.get("created_utc") == AMENDMENT_V5_CREATED_UTC,
        "V5 amendment envelope mismatch",
    )
    require_created_utc(payload["created_utc"], label="V5 amendment")
    _require_exact_tree(payload.get("bound_v3_amendment"), amendment_identity(), label="V5 bound V3 amendment")
    _require_exact_tree(payload.get("bound_v4_amendment"), amendment_v4_identity(), label="V5 bound V4 amendment")
    _require_exact_tree(payload.get("bound_v3_failure_incident"), incident_identity(), label="V5 bound incident")
    _require(
        _utc_order_value(v3["created_utc"], label="V3 amendment")
        < _utc_order_value(AMENDMENT_V4_CREATED_UTC, label="V4 amendment")
        < _utc_order_value(INCIDENT_CREATED_UTC, label="V3 incident")
        < _utc_order_value(AMENDMENT_V5_CREATED_UTC, label="V5 amendment"),
        "V3/V4/incident/V5 chronology mismatch",
    )
    _require_exact_tree(
        payload.get("code_and_test_paths_exact"),
        CODE_ROLE_PATHS,
        label="V5 exact code/test paths",
    )
    correction = payload.get("correction_scope")
    _require(
        isinstance(correction, Mapping)
        and correction.get("parsed_v3_v4_pointer_replacement_count") == 0
        and type(correction["parsed_v3_v4_pointer_replacement_count"]) is int
        and correction.get("logical_value_reads_before_failure") == 0
        and correction.get("fit_units_before_failure") == 0
        and correction.get("scope_expansion_forbidden") is True,
        "V5 correction scope mismatch",
    )
    controls = payload.get("control_contract")
    _require(isinstance(controls, Mapping), "V5 control contract absent")
    expected_paths = {
        "authorization": AUTHORIZATION_RELATIVE,
        "code_seal": CODE_SEAL_RELATIVE,
        "independent_go": GO_RELATIVE,
        "independent_postrun": POSTRUN_RELATIVE,
        "independent_review": REVIEW_RELATIVE,
    }
    _require_exact_tree(controls.get("control_paths_exact"), expected_paths, label="V5 control paths")
    _require_exact_tree(controls.get("output_namespace_exact"), OUTPUT_NAMESPACE, label="V5 control namespace")
    _require_exact_tree(controls.get("required_next_controls_exact"), REQUIRED_NEXT_CONTROLS, label="V5 next controls")
    _require_exact_tree(controls.get("required_review_exact"), REQUIRED_REVIEW, label="V5 required review")
    _require_exact_tree(controls.get("required_go_exact"), REQUIRED_GO, label="V5 required GO")
    _require_exact_tree(controls.get("single_attempt_exact"), SINGLE_ATTEMPT, label="V5 single attempt")
    _require(
        controls.get("control_schema_version_exact") == 5
        and type(controls["control_schema_version_exact"]) is int
        and controls.get("control_top_level_key_counts_exact")
        == {"AUTHORIZATION": 17, "CODE_SEAL": 13, "GO": 17, "POSTRUN": 18, "REVIEW": 16},
        "V5 control schema/count mismatch",
    )
    top_by_kind = controls.get("control_top_level_keys_exact")
    for contract_name, kind in (
        ("CODE_SEAL", "code_seal"),
        ("AUTHORIZATION", "authorization"),
        ("REVIEW", "review"),
        ("GO", "go"),
        ("POSTRUN", "postrun"),
    ):
        observed_keys = top_by_kind.get(contract_name) if isinstance(top_by_kind, Mapping) else None
        _require(
            isinstance(observed_keys, list)
            and len(observed_keys) == len(CONTROL_SPECS[kind]["keys"])
            and set(observed_keys) == set(CONTROL_SPECS[kind]["keys"]),
            f"V5 {contract_name} top key contract mismatch",
        )
    auditor = controls.get("auditor_stdout_contract")
    runner = controls.get("runner_stdout_contract")
    _require(
        isinstance(auditor, Mapping)
        and auditor.get("top_key_count") == 18
        and len(auditor.get("top_exact_keys", ())) == 18
        and set(auditor["top_exact_keys"]) == AUDITOR_STDOUT_KEYS
        and isinstance(runner, Mapping)
        and runner.get("top_key_count") == 11
        and len(runner.get("top_exact_keys", ())) == 11
        and set(runner["top_exact_keys"]) == RUNNER_STDOUT_KEYS,
        "V5 runner/auditor stdout contract mismatch",
    )
    output = payload.get("output_contract")
    _require(isinstance(output, Mapping), "V5 output contract absent")
    _require_exact_tree(output.get("output_namespace_exact"), OUTPUT_NAMESPACE, label="V5 output namespace")
    _require(output.get("output_root") == OUTPUT_ROOT_RELATIVE and output.get("runner_transaction_root") == TRANSACTION_ROOT_RELATIVE, "V5 output root mismatch")
    _require_exact_tree(output.get("runner_staged_paths_exact"), RUNNER_STAGED_RELATIVES, label="V5 runner staged paths")
    _require(output.get("final_publication_order_positive_exact") == list(FINAL_PUBLICATION_ORDER), "V5 positive publication order mismatch")
    _require(output.get("final_publication_order_negative_exact") == list(FINAL_PUBLICATION_ORDER), "V5 negative publication order mismatch")
    _require(output.get("completed_positive_paths_exact") == [f"{OUTPUT_ROOT_RELATIVE}{name}" for name in EXACT11_BASENAMES], "V5 positive exact11 mismatch")
    _require(output.get("completed_negative_or_ambiguity_paths_exact") == [f"{OUTPUT_ROOT_RELATIVE}{name}" for name in EXACT11_BASENAMES], "V5 negative exact11 mismatch")
    _require(output.get("single_v5_attempt_only") is True and output.get("create_new_exclusive_no_overwrite") is True and output.get("manifest_last_commit_marker") is True, "V5 publication flags mismatch")
    expected_aliases = {
        name: {"final_path": spec["final"], "original_source_path": spec["source"]}
        for name, spec in ALIAS_SPECS.items()
    }
    _require_exact_tree(output.get("aliases_exact"), expected_aliases, label="V5 original-source aliases")
    _require_exact_tree(
        output.get("versioned_evidence_basename_mapping_exact"),
        {
            "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3.parquet": "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V5.parquet",
            "TARGET_FREE_DISTRIBUTION_STABILITY_V3.json": "TARGET_FREE_DISTRIBUTION_STABILITY_V5.json",
            "TARGET_FREE_DUPLICATE_OOF_V3.parquet": "TARGET_FREE_DUPLICATE_OOF_V5.parquet",
            "TARGET_FREE_FIT_LEDGER_V3.json": "TARGET_FREE_FIT_LEDGER_V5.json",
        },
        label="V5 versioned evidence mapping",
    )
    final_binding = payload.get("final_lock_manifest_binding")
    _require(
        isinstance(final_binding, Mapping)
        and final_binding.get("superseding_exact_key_count") == 5
        and final_binding.get("superseding_exact_keys") == list(CODE_CONFIG_RUNTIME_KEYS)
        and final_binding.get("lock_generation_literals", {}).get("schema_version") == 5
        and final_binding.get("lock_generation_literals", {}).get("artifact_type") == LOCK_ARTIFACT_TYPE
        and final_binding.get("lock_generation_literals", {}).get("status_by_decision_status_exact") == LOCK_STATUS_BY_DECISION_STATUS
        and final_binding.get("manifest_generation_literals", {}).get("schema_version") == 5
        and final_binding.get("manifest_generation_literals", {}).get("artifact_type") == MANIFEST_ARTIFACT_TYPE
        and final_binding.get("manifest_generation_literals", {}).get("status") == MANIFEST_STATUS,
        "V5 final lock/manifest binding mismatch",
    )
    serialization = payload.get("runtime_and_serialization_preservation")
    _require(
        isinstance(serialization, Mapping)
        and serialization.get("v4_parquet_exact12_kwargs_verbatim")
        == V4_EXPLICIT_WRITE_TABLE_KWARGS
        and serialization.get("v4_csv_boundary_grammar_verbatim")
        == {
            "boundary_equality_flag_order_source": "component_decision_truth_table.continuous_cutoffs_exact",
            "boundary_equality_flags_none_representation": "EMPTY_FIELD",
            "boundary_equality_flags_present_representation": "SEMICOLON_JOINED_ACTIVE_CONTINUOUS_CUTOFF_NAMES",
            "corrected_v3_literal": V4_CSV_BOUNDARY_LITERAL,
            "csv_boolean_cells_exact_values": ["TRUE", "FALSE"],
            "duplicate_or_inactive_cutoff_names_forbidden": True,
        }
        and serialization.get("source_input_cast_or_schema_coercion_allowed") is False
        and serialization.get("output_arrow_schemas_unchanged_except_v5_filenames") is True
        and serialization.get("input_large_string_footer_types_do_not_change_output_string_types") is True,
        "V5 runtime/serialization preservation mismatch",
    )
    v4_parquet = v4.get("corrected_parquet_serialization")
    _require(
        isinstance(v4_parquet, Mapping)
        and serialization["v4_parquet_exact12_kwargs_verbatim"]
        == v4_parquet["explicit_write_table_kwargs_exact"],
        "V5/V4 exact12 write_table crosslink mismatch",
    )
    _require_exact_tree(payload.get("execution_budget"), EXECUTION_BUDGET, label="V5 execution budget")
    authority = payload.get("authority_scope")
    _require(
        isinstance(authority, Mapping)
        and authority.get("authorized_code_and_test_paths") == list(CODE_ROLE_PATHS.values())
        and authority.get("code_and_test_implementation_authorized") is True
        and authority.get("documentary_amendment") is True
        and all(
            value is False
            for key, value in authority.items()
            if key not in {"authorized_code_and_test_paths", "code_and_test_implementation_authorized", "documentary_amendment", "separate_code_seal_review_and_go_required"}
        )
        and authority.get("separate_code_seal_review_and_go_required") is True,
        "V5 documentary authority mismatch",
    )
    _require(
        isinstance(payload.get("prohibitions"), Mapping)
        and len(payload["prohibitions"]) == 36
        and all(value is True for value in payload["prohibitions"].values()),
        "V5 prohibitions mismatch",
    )
    return dict(payload)


def validate_amendment(artifact_root: Path) -> dict[str, Any]:
    v3 = _validate_v3_amendment(artifact_root)
    v4 = validate_amendment_v4(artifact_root, v3)
    incident = validate_v3_failure_incident(artifact_root)
    validate_amendment_v5(artifact_root, v3, v4, incident)
    effective = copy.deepcopy(v3)
    serialization = effective["output_contract"]["output_serialization_exact"]
    serialization["parquet"] = V4_PARQUET_LITERAL
    serialization["csv_boundary_flags"] = V4_CSV_BOUNDARY_LITERAL
    _require(
        serialization["parquet"] != v3["output_contract"]["output_serialization_exact"]["parquet"]
        and serialization["csv_boundary_flags"]
        != v3["output_contract"]["output_serialization_exact"]["csv_boundary_flags"],
        "V3+V4 overlay did not change exactly the two serialization literals",
    )
    return effective


def validate_sealer_runtime_identity(
    amendment: Mapping[str, Any],
) -> dict[str, str]:
    runtime = amendment["runtime_identity"]
    expected_executable = str(runtime["python_executable"]).replace("\\", "/")
    _require(
        normalized_absolute(Path(sys.executable)) == expected_executable,
        "V3 sealer Python executable identity drift",
    )
    _require(
        sys.version == runtime["python_full_version"],
        "V3 sealer Python full-version identity drift",
    )
    observed_platform = platform.platform()
    _require(
        observed_platform == runtime["platform"],
        "V3 sealer platform identity drift",
    )
    return {
        "python_executable": normalized_absolute(Path(sys.executable)),
        "python_full_version": sys.version,
        "platform": observed_platform,
    }


def collect_code_identities(repo_root: Path) -> dict[str, dict[str, Any]]:
    repo_root = Path(os.path.abspath(repo_root))
    missing = [relative for relative in CODE_ROLE_PATHS.values() if not (repo_root / relative).is_file()]
    if missing:
        raise DynamicIdentityPending(f"code/test identities remain dynamic; missing: {missing}")
    return {
        role: file_identity(
            path_under(repo_root, relative, label=f"code role {role}"),
            relative_to=None,
        )
        for role, relative in CODE_ROLE_PATHS.items()
    }


def validate_code_identities(
    records: Any, repo_root: Path, *, rehash: bool = True
) -> dict[str, dict[str, Any]]:
    _require(isinstance(records, Mapping), "code_identities is not an object")
    _require(set(records) == set(CODE_ROLE_PATHS), "code identity role set mismatch")
    result: dict[str, dict[str, Any]] = {}
    for role, relative in CODE_ROLE_PATHS.items():
        expected = path_under(repo_root, relative, label=f"code identity {role}")
        result[role] = validate_identity_record(
            records[role], expected, label=f"code identity {role}", relative_to=None, rehash=rehash
        )
    return result


def expected_required_commands(
    artifact_root: Path, amendment: Mapping[str, Any]
) -> dict[str, list[str]]:
    runtime = amendment["runtime_identity"]
    python = str(Path(str(runtime["python_executable"])))
    root = Path(os.path.abspath(artifact_root))
    root_text = str(root)
    authorization = str(path_under(root, AUTHORIZATION_RELATIVE, label="authorization command"))
    go = str(path_under(root, GO_RELATIVE, label="GO command"))
    postrun = str(path_under(root, POSTRUN_RELATIVE, label="postrun command"))
    common = [
        "--root", root_text,
        "--authorization", authorization,
        "--independent-go", go,
    ]
    return {
        "runner": [python, "-B", "-m", "scripts.run_noaa_gfs_target_free_duplicate_v5", *common],
        "auditor": [python, "-B", "-m", "scripts.audit_noaa_gfs_target_free_duplicate_v5", *common],
        "sealer": [python, "-B", "-m", "scripts.seal_noaa_gfs_target_free_duplicate_v5", *common, "--postrun-pass", postrun],
    }


def expected_test_commands(
    repo_root: Path, amendment: Mapping[str, Any]
) -> dict[str, list[str]]:
    python = str(Path(str(amendment["runtime_identity"]["python_executable"])))
    repo = Path(os.path.abspath(repo_root))
    return {
        role: [
            python,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(path_under(repo, test_path, label=f"{role} test command")),
        ]
        for role, test_path in TEST_ROLE_PATHS.items()
    }


def validate_test_evidence(
    evidence: Any,
    *,
    repo_root: Path,
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(evidence, Mapping), "test_evidence is not an object")
    _require(set(evidence) == TEST_EVIDENCE_KEYS, "test_evidence schema mismatch")
    commands = evidence.get("commands")
    results = evidence.get("results")
    _require(isinstance(commands, Mapping) and set(commands) == set(TEST_ROLE_PATHS), "test command role set mismatch")
    _require(isinstance(results, Mapping) and set(results) == set(TEST_ROLE_PATHS), "test result role set mismatch")
    _require(commands == expected_test_commands(repo_root, amendment), "test evidence commands mismatch")
    for role, expected_test in TEST_ROLE_PATHS.items():
        result = results[role]
        _require(isinstance(result, Mapping) and set(result) == TEST_RESULT_KEYS, f"test result schema mismatch: {role}")
        _require(result.get("test_file") == expected_test, f"test result file mismatch: {role}")
        _require(
            isinstance(result.get("exit_code"), int)
            and not isinstance(result["exit_code"], bool)
            and result["exit_code"] == 0,
            f"test result exit code failed/mismatched: {role}",
        )
        for key in ("passed", "skipped", "deselected"):
            _require(isinstance(result.get(key), int) and not isinstance(result[key], bool) and int(result[key]) >= 0, f"test result count invalid: {role}/{key}")
        _require(result["passed"] > 0, f"test result has no passing test: {role}")
        summary = result.get("summary")
        summary_match = (
            PYTEST_SUMMARY.fullmatch(summary)
            if isinstance(summary, str)
            else None
        )
        _require(summary_match is not None, f"test summary invalid: {role}")
        assert isinstance(summary, str)
        assert summary_match is not None
        summary_counts = {
            "passed": int(summary_match.group("passed")),
            "skipped": int(summary_match.group("skipped") or 0),
            "deselected": int(summary_match.group("deselected") or 0),
        }
        _require(
            all(result[key] == summary_counts[key] for key in summary_counts),
            f"test summary/count fields disagree: {role}",
        )
        if summary_match.group("hours") is not None:
            suffix_seconds = (
                int(summary_match.group("hours")) * 3_600
                + int(summary_match.group("minutes")) * 60
                + int(summary_match.group("seconds"))
            )
            displayed_whole = int(summary_match.group("duration_whole"))
            fraction = summary_match.group("duration_fraction")
            rounded_up_to_zero_fraction = (
                fraction is not None
                and set(fraction) == {"0"}
                and suffix_seconds == displayed_whole - 1
            )
            _require(
                suffix_seconds >= 60
                and (
                    suffix_seconds == displayed_whole
                    or rounded_up_to_zero_fraction
                ),
                f"test summary duration suffix disagrees: {role}",
            )
    for key in (
        "all_exit_codes_zero", "all_expected_test_files_covered", "network_denied",
        "pycache_disabled", "code_identities_revalidated",
    ):
        _require(evidence.get(key) is True, f"test evidence false: {key}")
    require_created_utc(evidence.get("completed_utc"), label="test evidence")
    return dict(evidence)


def validate_output_identity_record(
    record: Any,
    *,
    expected_relative: str,
    expected_format: str,
    expected_rows: int,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    _require(isinstance(record, Mapping), "output identity is not an object")
    _require(set(record) == OUTPUT_IDENTITY_KEYS, f"output identity schema mismatch: {expected_relative}")
    _require(record.get("path") == expected_relative, f"output identity path mismatch: {expected_relative}")
    _require(record.get("format") == expected_format, f"output format mismatch: {expected_relative}")
    _require(record.get("row_count") == expected_rows, f"output row count mismatch: {expected_relative}")
    _require(isinstance(record.get("size_bytes"), int) and not isinstance(record["size_bytes"], bool) and int(record["size_bytes"]) > 0, f"output size invalid: {expected_relative}")
    _require(isinstance(record.get("row_count"), int) and not isinstance(record["row_count"], bool), f"output row count type invalid: {expected_relative}")
    for key in ("sha256", "logical_sha256"):
        _require(isinstance(record.get(key), str) and LOWER_SHA256.fullmatch(str(record[key])) is not None, f"output {key} invalid: {expected_relative}")
    if expected_format in {"CSV", "JSON"}:
        _require(record["logical_sha256"] == record["sha256"], f"text logical/physical identity mismatch: {expected_relative}")
    if artifact_root is not None:
        path = path_under(artifact_root, expected_relative, label="staged output")
        observed = file_identity(path, relative_to=artifact_root)
        _require(observed["size_bytes"] == record["size_bytes"] and observed["sha256"] == record["sha256"], f"staged physical identity mismatch: {expected_relative}")
    return dict(record)


def validate_staged_output_identities(
    value: Any, *, artifact_root: Path | None = None
) -> dict[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == {"aliases", "runner_staged_outputs"}, "staged_output_identities schema mismatch")
    aliases = value["aliases"]
    runner = value["runner_staged_outputs"]
    _require(isinstance(aliases, Mapping) and set(aliases) == set(ALIAS_ORDER), "alias identity role set mismatch")
    for name, spec in ALIAS_SPECS.items():
        record = aliases[name]
        expected_path = (
            path_under(artifact_root, spec["final"], label=f"alias {name}")
            if artifact_root is not None
            else Path(spec["final"])
        )
        if artifact_root is None:
            _require(isinstance(record, Mapping) and set(record) == FILE_IDENTITY_KEYS, f"alias identity schema mismatch: {name}")
            _require(record.get("path") == spec["final"], f"alias path mismatch: {name}")
        else:
            validate_identity_record(record, expected_path, label=f"alias {name}", relative_to=artifact_root, rehash=True)
        _require(record.get("size_bytes") == spec["size_bytes"] and record.get("sha256") == spec["sha256"], f"alias frozen identity mismatch: {name}")
    _require(isinstance(runner, Mapping) and set(runner) == set(RUNNER_PUBLICATION_ORDER), "runner staged identity role set mismatch")
    for name in RUNNER_PUBLICATION_ORDER:
        validate_output_identity_record(
            runner[name],
            expected_relative=RUNNER_STAGED_RELATIVES[name],
            expected_format=EXPECTED_RUNNER_FORMATS[name],
            expected_rows=EXPECTED_RUNNER_ROWS[name],
            artifact_root=artifact_root,
        )
    return {"aliases": dict(aliases), "runner_staged_outputs": dict(runner)}


def validate_control_payload(
    kind: str,
    payload: Any,
    *,
    amendment: Mapping[str, Any],
    artifact_root: Path | None = None,
    repo_root: Path | None = None,
    rehash_code: bool = False,
) -> dict[str, Any]:
    _require(kind in CONTROL_SPECS, f"unknown control kind: {kind}")
    spec = CONTROL_SPECS[kind]
    _require(isinstance(payload, Mapping), f"{kind} control is not an object")
    _require(set(payload) == spec["keys"], f"{kind} top-level schema mismatch")
    _require(
        isinstance(payload.get("schema_version"), int)
        and not isinstance(payload["schema_version"], bool)
        and payload["schema_version"] == 5,
        f"{kind} schema version mismatch",
    )
    _require(payload.get("artifact_type") == spec["artifact_type"], f"{kind} artifact type mismatch")
    _require(payload.get("status") == spec["status"], f"{kind} status mismatch")
    _require(payload.get("publisher") == spec["publisher"], f"{kind} publisher mismatch")
    _require_exact_tree(
        payload.get("authority_scope"), spec["authority_scope"],
        label=f"{kind} authority scope",
    )
    _require_exact_tree(
        payload.get("amendment"), amendment_identity(),
        label=f"{kind} amendment identity",
    )
    _require_exact_tree(
        payload.get("amendment_v4"), amendment_v4_identity(),
        label=f"{kind} V4 amendment identity",
    )
    _require_exact_tree(
        payload.get("amendment_v5"), amendment_v5_identity(),
        label=f"{kind} V5 amendment identity",
    )
    if kind != "postrun":
        _require_exact_tree(
            payload.get("output_namespace"), OUTPUT_NAMESPACE,
            label=f"{kind} output namespace",
        )
    require_created_utc(payload.get("created_utc"), label=kind)
    _require(
        _utc_order_value(AMENDMENT_V5_CREATED_UTC, label="V5 amendment")
        < _utc_order_value(payload["created_utc"], label=kind),
        f"{kind} was created before the V5 documentary amendment",
    )
    if repo_root is None:
        _require(isinstance(payload.get("code_identities"), Mapping) and set(payload["code_identities"]) == set(CODE_ROLE_PATHS), f"{kind} code identities role mismatch")
        for role, record in payload["code_identities"].items():
            _require(isinstance(record, Mapping) and set(record) == FILE_IDENTITY_KEYS, f"{kind} code identity schema mismatch: {role}")
    else:
        validate_code_identities(payload["code_identities"], repo_root, rehash=rehash_code)

    if kind == "code_seal":
        validate_test_evidence(
            payload["test_evidence"],
            repo_root=repo_root or REPO,
            amendment=amendment,
        )
        _require(
            _utc_order_value(AMENDMENT_V5_CREATED_UTC, label="V5 amendment")
            < _utc_order_value(
                payload["test_evidence"]["completed_utc"],
                label="V5 test evidence",
            )
            <= _utc_order_value(payload["created_utc"], label="V5 code seal"),
            "V5 code-seal/test-evidence chronology mismatch",
        )
        _require_exact_tree(
            payload["required_next_controls"], REQUIRED_NEXT_CONTROLS,
            label="code-seal next-control paths",
        )
    elif kind == "authorization":
        _require_exact_tree(
            payload["execution_budget"], EXECUTION_BUDGET,
            label="authorization execution budget",
        )
        root = artifact_root or Path(str(amendment["precedence"]["artifact_root_absolute"]))
        _require_exact_tree(
            payload["required_commands"], expected_required_commands(root, amendment),
            label="authorization required commands",
        )
        _require_exact_tree(
            payload["required_review"], REQUIRED_REVIEW,
            label="authorization review path",
        )
        _require_exact_tree(
            payload["required_go"], REQUIRED_GO,
            label="authorization GO path",
        )
        require_attempt_id(payload.get("attempt_id"), label="authorization")
        _require(isinstance(payload.get("code_seal"), Mapping) and set(payload["code_seal"]) == FILE_IDENTITY_KEYS, "authorization code-seal identity schema mismatch")
    elif kind == "review":
        require_attempt_id(payload.get("attempt_id"), label="review")
        _require(payload.get("verdict") == "PASS", "review verdict mismatch")
        checks = payload.get("independent_checks")
        _require(isinstance(checks, Mapping) and set(checks) == REVIEW_CHECK_KEYS and all(value is True for value in checks.values()), "review independent checks mismatch")
        for field in ("code_seal", "authorization"):
            _require(isinstance(payload.get(field), Mapping) and set(payload[field]) == FILE_IDENTITY_KEYS, f"review {field} identity schema mismatch")
    elif kind == "go":
        require_attempt_id(payload.get("attempt_id"), label="GO")
        _require_exact_tree(
            payload.get("single_attempt"), SINGLE_ATTEMPT,
            label="GO single-attempt contract",
        )
        root = artifact_root or Path(str(amendment["precedence"]["artifact_root_absolute"]))
        _require_exact_tree(
            payload.get("required_commands"), expected_required_commands(root, amendment),
            label="GO required commands",
        )
        for field in ("code_seal", "authorization", "independent_review"):
            _require(isinstance(payload.get(field), Mapping) and set(payload[field]) == FILE_IDENTITY_KEYS, f"GO {field} identity schema mismatch")
    elif kind == "postrun":
        require_attempt_id(payload.get("attempt_id"), label="postrun")
        _require(payload.get("verdict") == "PASS", "postrun verdict mismatch")
        for field in ("code_seal", "authorization", "independent_review", "independent_go"):
            _require(isinstance(payload.get(field), Mapping) and set(payload[field]) == FILE_IDENTITY_KEYS, f"postrun {field} identity schema mismatch")
        validate_staged_output_identities(
            payload["staged_output_identities"], artifact_root=artifact_root
        )
        validate_audit_result(
            payload["audit_result"],
            payload,
            amendment=amendment,
            artifact_root=artifact_root,
        )
    return dict(payload)


def validate_auditor_stdout(
    payload: Any,
    *,
    attempt_id: str,
    amendment: Mapping[str, Any],
    amendment_record: Mapping[str, Any],
    amendment_v4_record: Mapping[str, Any],
    amendment_v5_record: Mapping[str, Any],
    authorization_record: Mapping[str, Any],
    go_record: Mapping[str, Any],
    code_identities: Mapping[str, Any],
    staged_output_identities: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(payload, Mapping) and set(payload) == AUDITOR_STDOUT_KEYS, "auditor stdout schema mismatch")
    _require(
        isinstance(payload.get("schema_version"), int)
        and not isinstance(payload["schema_version"], bool)
        and payload["schema_version"] == 5,
        "auditor stdout schema version mismatch",
    )
    _require(payload.get("artifact_type") == "TARGET_FREE_DUPLICATE_NO_FIT_POSTRUN_AUDIT_V5", "auditor stdout artifact type mismatch")
    _require(payload.get("status") == "PASS_V5_NO_FIT_POSTRUN_AUDIT", "auditor stdout status mismatch")
    require_attempt_id(payload.get("attempt_id"), label="auditor stdout")
    _require(payload.get("attempt_id") == attempt_id, "auditor stdout attempt mismatch")
    _require_exact_tree(
        payload.get("amendment"), amendment_record,
        label="auditor stdout amendment",
    )
    _require_exact_tree(
        payload.get("amendment_v4"), amendment_v4_record,
        label="auditor stdout V4 amendment",
    )
    _require_exact_tree(
        payload.get("amendment_v5"), amendment_v5_record,
        label="auditor stdout V5 amendment",
    )
    _require_exact_tree(
        payload.get("authorization"), authorization_record,
        label="auditor stdout authorization",
    )
    _require_exact_tree(
        payload.get("independent_go"), go_record,
        label="auditor stdout GO",
    )
    _require_exact_tree(
        payload.get("code_identities"), code_identities,
        label="auditor stdout code identities",
    )
    _require_exact_tree(
        payload.get("staged_output_identities"), staged_output_identities,
        label="auditor stdout staged identities",
    )
    for field, message in (
        ("files_written", "no-fit auditor wrote files"),
        ("network_requests", "no-fit auditor used network"),
        ("models_fit", "no-fit auditor fit a model"),
    ):
        _require(
            isinstance(payload.get(field), int)
            and not isinstance(payload[field], bool)
            and payload[field] == 0,
            message,
        )
    _require_exact_tree(
        payload.get("forbidden_reads"), AUDITOR_FORBIDDEN_READS,
        label="no-fit auditor forbidden-read facts",
    )
    _require_exact_tree(
        payload.get("output_counts"), AUDITOR_OUTPUT_COUNTS,
        label="auditor stdout output_counts",
    )
    decision = payload.get("decision")
    _require(isinstance(decision, Mapping) and set(decision) == AUDITOR_DECISION_KEYS, "auditor stdout decision schema mismatch")
    statuses = amendment.get("output_contract", {}).get(
        "decision_status_literals_exact"
    )
    _require(isinstance(statuses, Mapping), "auditor decision status contract absent")
    _require(decision.get("status") in set(statuses.values()), "auditor decision status mismatch")
    _require(isinstance(decision.get("global_ambiguity"), bool), "auditor global_ambiguity is not boolean")
    selected = decision.get("selected_family")
    if decision["status"] == statuses.get("positive"):
        families = amendment.get("family_decision_and_salvage", {}).get(
            "fixed_priority_if_both_pass"
        )
        _require(
            isinstance(families, list)
            and selected in families
            and decision["global_ambiguity"] is False,
            "auditor positive decision family/ambiguity mismatch",
        )
    elif decision["status"] == statuses.get("global_ambiguity"):
        _require(
            selected is None and decision["global_ambiguity"] is True,
            "auditor global-ambiguity decision mismatch",
        )
    else:
        _require(
            decision["status"] == statuses.get("clear_no_family")
            and selected is None
            and decision["global_ambiguity"] is False,
            "auditor clear-no-family decision mismatch",
        )
    _require_exact_tree(
        payload.get("poststate"), AUDITOR_POSTSTATE,
        label="auditor stdout poststate",
    )
    return dict(payload)


def _validate_process_capture(
    result: Any,
    *,
    expected_command: Sequence[str],
    label: str,
) -> Mapping[str, Any]:
    _require(
        isinstance(result, Mapping) and set(result) == PROCESS_CAPTURE_KEYS,
        f"{label} process capture schema mismatch",
    )
    _require(
        result.get("command") == list(expected_command),
        f"{label} command mismatch",
    )
    _require(
        isinstance(result.get("exit_code"), int)
        and not isinstance(result["exit_code"], bool)
        and result["exit_code"] == 0,
        f"{label} exit code mismatch",
    )
    _require(
        isinstance(result.get("stdout_size_bytes"), int)
        and not isinstance(result["stdout_size_bytes"], bool)
        and result["stdout_size_bytes"] > 0,
        f"{label} stdout size type mismatch",
    )
    _require(
        isinstance(result.get("stdout_sha256"), str)
        and LOWER_SHA256.fullmatch(result["stdout_sha256"]) is not None,
        f"{label} stdout SHA type mismatch",
    )
    _require(
        isinstance(result.get("stderr_size_bytes"), int)
        and not isinstance(result["stderr_size_bytes"], bool)
        and result["stderr_size_bytes"] == 0
        and result.get("stderr_sha256") == EMPTY_SHA256,
        f"{label} stderr is nonempty",
    )
    _require(
        result.get("full_stdout_identity_instrumented") is True,
        f"{label} full stdout identity was not instrumented",
    )
    wall = result.get("wall_seconds_observed")
    _require(
        isinstance(wall, (int, float))
        and not isinstance(wall, bool)
        and math.isfinite(float(wall))
        and float(wall) >= 0.0,
        f"{label} wall time invalid",
    )
    return result


def _expected_threadpool_observation(
    amendment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    native = amendment.get("runtime_identity", {}).get("native_libraries_exact")
    _require(
        isinstance(native, Mapping) and set(native) == set(NATIVE_LIBRARY_ORDER),
        "runner native-library set differs from the frozen runtime identity",
    )
    observations: list[dict[str, Any]] = []
    for name in NATIVE_LIBRARY_ORDER:
        item = native[name]
        _require(isinstance(item, Mapping), f"runner native metadata absent: {name}")
        observations.append(
            {
                "name": name,
                "path": item.get("path"),
                "user_api": item.get("user_api"),
                "internal_api": item.get("internal_api"),
                "prefix": item.get("prefix"),
                "version": item.get("version"),
                "threading_layer": item.get("threading_layer"),
                "architecture": item.get("architecture"),
                "num_threads": 1,
            }
        )
    return observations


def _validate_threadpool_event(
    value: Any,
    *,
    phase: str,
    observations: Sequence[Mapping[str, Any]],
    event_count: int,
    label: str,
) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping) and set(value) == THREADPOOL_EVENT_KEYS,
        f"{label} event schema mismatch",
    )
    _require(value.get("phase") == phase, f"{label} phase mismatch")
    _require(
        isinstance(value.get("event_ordinal"), int)
        and not isinstance(value["event_ordinal"], bool)
        and 1 <= value["event_ordinal"] <= event_count,
        f"{label} event ordinal mismatch",
    )
    _require(
        isinstance(value.get("label"), str) and bool(value["label"]),
        f"{label} event label mismatch",
    )
    _require_exact_tree(
        value.get("threadpools"), list(observations),
        label=f"{label} native threadpool observation",
    )
    return dict(value)


def validate_threadpool_evidence_structure(
    evidence: Any,
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        isinstance(evidence, Mapping) and set(evidence) == THREADPOOL_EVIDENCE_KEYS,
        "runner threadpool evidence schema mismatch",
    )
    _require_exact_tree(
        evidence.get("capture_phases"), list(THREADPOOL_PHASES),
        label="runner threadpool phase order",
    )
    counts = evidence.get("capture_counts")
    _require(
        isinstance(counts, Mapping) and set(counts) == set(THREADPOOL_PHASES),
        "runner threadpool capture-count schema mismatch",
    )
    _require(
        all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for value in counts.values()
        ),
        "runner threadpool capture count type mismatch",
    )
    inside = counts["INSIDE_CONTEXT"]
    _require(
        counts["BEFORE_EXPLICIT_CONTEXT"] == 1
        and counts["AFTER_CONTEXT"] == inside,
        "runner threadpool phase-count relation mismatch",
    )
    event_count = evidence.get("event_count")
    _require(
        isinstance(event_count, int)
        and not isinstance(event_count, bool)
        and event_count == 1 + 2 * inside,
        "runner threadpool event-count relation mismatch",
    )
    _require(
        isinstance(evidence.get("event_chain_sha256"), str)
        and LOWER_SHA256.fullmatch(evidence["event_chain_sha256"]) is not None,
        "runner threadpool event-chain SHA mismatch",
    )
    observations = _expected_threadpool_observation(amendment)
    expected_before = {
        "event_ordinal": 1,
        "phase": "BEFORE_EXPLICIT_CONTEXT",
        "label": THREADPOOL_BEFORE_LABEL,
        "threadpools": observations,
    }
    for boundary_name in (
        "first_observation_by_phase",
        "last_observation_by_phase",
    ):
        boundaries = evidence.get(boundary_name)
        _require(
            isinstance(boundaries, Mapping)
            and set(boundaries) == set(THREADPOOL_PHASES),
            f"runner threadpool {boundary_name} schema mismatch",
        )
        _require_exact_tree(
            boundaries["BEFORE_EXPLICIT_CONTEXT"], expected_before,
            label=f"runner threadpool {boundary_name} before event",
        )
        for phase in THREADPOOL_PHASES[1:]:
            event = boundaries[phase]
            if inside == 0:
                _require(
                    event is None,
                    f"runner zero-fit {boundary_name}/{phase} must be null",
                )
            else:
                _validate_threadpool_event(
                    event,
                    phase=phase,
                    observations=observations,
                    event_count=event_count,
                    label=f"runner threadpool {boundary_name}/{phase}",
                )
    return dict(evidence)


def _expected_threadpool_evidence_from_ledger(
    fit_ledger: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    observations = _expected_threadpool_observation(amendment)
    slots = fit_ledger["unit_slots"]
    responses = tuple(amendment["planned_fit_budget"]["response_order"])
    primary = slots[: FIT_LEDGER_PLANNED_COUNTS["primary_model_units"]]
    affine = slots[FIT_LEDGER_PLANNED_COUNTS["primary_model_units"] :]
    chain = hashlib.sha256()
    ordinal = 0
    counts = {phase: 0 for phase in THREADPOOL_PHASES}
    first: dict[str, Any] = {phase: None for phase in THREADPOOL_PHASES}
    last: dict[str, Any] = {phase: None for phase in THREADPOOL_PHASES}

    def capture(phase: str, event_label: str) -> None:
        nonlocal ordinal
        ordinal += 1
        counts[phase] += 1
        event = {
            "event_ordinal": ordinal,
            "phase": phase,
            "label": event_label,
            "threadpools": observations,
        }
        encoded = compact_json_bytes(event)
        chain.update(len(encoded).to_bytes(8, "big"))
        chain.update(encoded)
        if first[phase] is None:
            first[phase] = event
        last[phase] = event

    capture("BEFORE_EXPLICIT_CONTEXT", THREADPOOL_BEFORE_LABEL)
    for response in responses:
        for slot in primary:
            if slot["response"] != response or slot["fit_completed"] is not True:
                continue
            base = f"{response}/{slot['fold']}/{slot['estimator_unit']}"
            for action in ("FIT", "PREDICT"):
                event_label = f"{base}/{action}"
                capture("INSIDE_CONTEXT", event_label)
                capture("AFTER_CONTEXT", event_label)
        for slot in affine:
            if slot["response"] != response or slot["fit_completed"] is not True:
                continue
            event_label = (
                f"{response}/{slot['predictor']}/{slot['fold']}/"
                "ANALYTIC_AFFINE/FIT_PREDICT"
            )
            capture("INSIDE_CONTEXT", event_label)
            capture("AFTER_CONTEXT", event_label)
    return {
        "capture_phases": list(THREADPOOL_PHASES),
        "capture_counts": counts,
        "event_count": ordinal,
        "event_chain_sha256": chain.hexdigest(),
        "first_observation_by_phase": first,
        "last_observation_by_phase": last,
    }


def validate_runner_stdout(
    payload: Any,
    *,
    attempt_id: str,
    amendment: Mapping[str, Any],
    staged_output_identities: Mapping[str, Any],
    expected_decision: Mapping[str, Any],
    fit_ledger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(
        isinstance(payload, Mapping) and set(payload) == RUNNER_STDOUT_KEYS,
        "runner stdout schema mismatch",
    )
    _require(
        isinstance(payload.get("schema_version"), int)
        and not isinstance(payload["schema_version"], bool)
        and payload["schema_version"] == 5,
        "runner stdout schema version mismatch",
    )
    _require(
        payload.get("artifact_type")
        == "TARGET_FREE_DUPLICATE_RUNNER_STAGED_RESULT_V5",
        "runner stdout artifact type mismatch",
    )
    _require(
        payload.get("status")
        == "RUNNER_STAGED_OUTPUTS_COMPLETE_PENDING_POSTRUN_AUDIT",
        "runner stdout status mismatch",
    )
    require_attempt_id(payload.get("attempt_id"), label="runner stdout")
    _require(payload.get("attempt_id") == attempt_id, "runner stdout attempt mismatch")
    _require_exact_tree(
        payload.get("decision"), expected_decision,
        label="runner stdout decision/auditor crosslink",
    )
    _require_exact_tree(
        payload.get("output_counts"), AUDITOR_OUTPUT_COUNTS,
        label="runner stdout output_counts",
    )
    runner_mapping = staged_output_identities.get("runner_staged_outputs")
    _require(
        isinstance(runner_mapping, Mapping)
        and set(runner_mapping) == set(RUNNER_PUBLICATION_ORDER),
        "runner stdout staged-identity source set mismatch",
    )
    _require_exact_tree(
        payload.get("staged_output_identities"),
        [runner_mapping[name] for name in RUNNER_PUBLICATION_ORDER],
        label="runner stdout staged7/postrun crosslink",
    )
    _require_exact_tree(
        payload.get("no_forbidden_access_attestation"),
        DECISION_FORBIDDEN_ACCESS_ATTESTATION,
        label="runner stdout forbidden-access attestation",
    )
    _require(
        payload.get("manifest_present") is False
        and payload.get("family_lock_present") is False,
        "runner stdout pre-final-publication state mismatch",
    )
    threadpool = validate_threadpool_evidence_structure(
        payload.get("threadpool_info_before_inside_after"), amendment
    )
    if fit_ledger is not None:
        validate_fit_ledger_payload(
            fit_ledger,
            amendment,
            expected_status=str(expected_decision["status"]),
        )
        expected_threadpool = _expected_threadpool_evidence_from_ledger(
            fit_ledger, amendment
        )
        _require_exact_tree(
            threadpool, expected_threadpool,
            label="runner threadpool full ledger-derived event chain",
        )
    return dict(payload)


def validate_runner_result(
    result: Any,
    postrun: Mapping[str, Any],
    *,
    amendment: Mapping[str, Any],
    artifact_root: Path,
    expected_decision: Mapping[str, Any],
    fit_ledger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    capture = _validate_process_capture(
        result,
        expected_command=expected_required_commands(artifact_root, amendment)[
            "runner"
        ],
        label="postrun runner",
    )
    stdout_payload = capture.get("stdout_payload")
    validate_runner_stdout(
        stdout_payload,
        attempt_id=str(postrun["attempt_id"]),
        amendment=amendment,
        staged_output_identities=postrun["staged_output_identities"],
        expected_decision=expected_decision,
        fit_ledger=fit_ledger,
    )
    encoded = pretty_json_bytes(stdout_payload)
    _require(
        capture.get("stdout_size_bytes") == len(encoded),
        "postrun runner stdout size mismatch",
    )
    _require(
        capture.get("stdout_sha256") == hashlib.sha256(encoded).hexdigest(),
        "postrun runner stdout SHA mismatch",
    )
    return dict(capture)


def validate_audit_result(
    result: Any,
    postrun: Mapping[str, Any],
    *,
    amendment: Mapping[str, Any],
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    _require(isinstance(result, Mapping) and set(result) == AUDIT_RESULT_KEYS, "postrun audit_result schema mismatch")
    root = artifact_root or Path(str(amendment["precedence"]["artifact_root_absolute"]))
    auditor_capture = {
        key: result[key] for key in PROCESS_CAPTURE_KEYS
    }
    _validate_process_capture(
        auditor_capture,
        expected_command=expected_required_commands(root, amendment)["auditor"],
        label="postrun auditor",
    )
    stdout_payload = auditor_capture.get("stdout_payload")
    validate_auditor_stdout(
        stdout_payload,
        attempt_id=str(postrun["attempt_id"]),
        amendment=amendment,
        amendment_record=postrun["amendment"],
        amendment_v4_record=postrun["amendment_v4"],
        amendment_v5_record=postrun["amendment_v5"],
        authorization_record=postrun["authorization"],
        go_record=postrun["independent_go"],
        code_identities=postrun["code_identities"],
        staged_output_identities=postrun["staged_output_identities"],
    )
    encoded = pretty_json_bytes(stdout_payload)
    _require(auditor_capture.get("stdout_size_bytes") == len(encoded), "postrun stdout size mismatch")
    _require(auditor_capture.get("stdout_sha256") == hashlib.sha256(encoded).hexdigest(), "postrun stdout SHA mismatch")
    validate_runner_result(
        result.get("runner_result"),
        postrun,
        amendment=amendment,
        artifact_root=root,
        expected_decision=stdout_payload["decision"],
    )
    return dict(result)


def _utc_order_value(value: Any, *, label: str) -> int:
    text = require_created_utc(value, label=label)
    parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = parsed - epoch
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def validate_actual_execution_argv(
    actual_orig_argv: Any,
    authorized_sealer_argv: Any,
) -> list[str]:
    """Bind this production sealer process to the frozen authorized argv."""

    _require(
        type(actual_orig_argv) is list
        and all(type(item) is str for item in actual_orig_argv),
        "actual sealer sys.orig_argv must be an exact list of strings",
    )
    _require(
        type(authorized_sealer_argv) is list
        and all(type(item) is str for item in authorized_sealer_argv),
        "authorized sealer argv must be an exact list of strings",
    )
    _require(
        _exact_tree_equal(actual_orig_argv, authorized_sealer_argv),
        "actual sealer sys.orig_argv differs from the frozen authorized command",
    )
    return list(actual_orig_argv)


def load_control_create_if_absent_identity(
    artifact_root: Path,
    relative: str,
    *,
    label: str,
    supplied_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one exact control path and bind its artifact-root-relative bytes."""

    root = Path(os.path.abspath(artifact_root))
    expected = path_under(root, relative, label=label)
    if supplied_path is not None:
        supplied = Path(os.path.abspath(supplied_path))
        _require(supplied == expected, f"{label} CLI path is not the frozen exact path")
    payload = strict_json_load(expected, label=label)
    return payload, file_identity(expected, relative_to=root)


def validate_historical_v3_failure_state(
    artifact_root: Path,
) -> dict[str, Any]:
    """Revalidate immutable V3 controls and the preserved aliases-only root."""

    root = Path(os.path.abspath(artifact_root))
    incident = validate_v3_failure_incident(root)
    expected_envelopes = {
        "code_seal": (
            "TARGET_FREE_DUPLICATE_CODE_SEAL_V3",
            "SEALED_V3_CODE_AND_TESTS_NO_EXECUTION_AUTHORITY",
        ),
        "authorization": (
            "TARGET_FREE_DUPLICATE_EXECUTION_AUTHORIZATION_V3",
            "AUTHORIZED_V3_SINGLE_ATTEMPT_PENDING_INDEPENDENT_REVIEW_AND_GO",
        ),
        "independent_review": (
            "TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_REVIEW_V3",
            "PASS_V3_EXECUTION_REVIEW_PENDING_INDEPENDENT_GO",
        ),
        "independent_go": (
            "TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_GO_V3",
            "GO_V3_SINGLE_TARGET_FREE_ATTEMPT",
        ),
    }
    historical: dict[str, Mapping[str, Any]] = {}
    for role, record in incident["control_identities"].items():
        path = path_under(root, record["path"], label=f"historical V3 {role}")
        payload = strict_json_load(path, label=f"historical V3 {role}")
        artifact_type, status = expected_envelopes[role]
        _require(
            payload.get("schema_version") == 3
            and type(payload["schema_version"]) is int
            and payload.get("artifact_type") == artifact_type
            and payload.get("status") == status,
            f"historical V3 control envelope mismatch: {role}",
        )
        require_created_utc(payload.get("created_utc"), label=f"historical V3 {role}")
        historical[role] = payload
    chronology = [
        _utc_order_value(
            _validate_v3_amendment(root)["created_utc"], label="V3 amendment"
        ),
        _utc_order_value(AMENDMENT_V4_CREATED_UTC, label="V4 amendment"),
        _utc_order_value(historical["code_seal"]["created_utc"], label="V3 code seal"),
        _utc_order_value(historical["authorization"]["created_utc"], label="V3 authorization"),
        _utc_order_value(historical["independent_review"]["created_utc"], label="V3 review"),
        _utc_order_value(historical["independent_go"]["created_utc"], label="V3 GO"),
        _utc_order_value(INCIDENT_CREATED_UTC, label="V3 incident"),
        _utc_order_value(AMENDMENT_V5_CREATED_UTC, label="V5 amendment"),
    ]
    _require(
        all(left < right for left, right in zip(chronology, chronology[1:])),
        "historical V3/V4/incident/V5 chronology mismatch",
    )

    failed_root = path_under(
        root, V3_OUTPUT_ROOT_RELATIVE.rstrip("/"), label="preserved V3 output root"
    )
    _require(
        failed_root.is_dir() and not _linklike(failed_root),
        "preserved V3 output root absent/link-like",
    )
    entries = {path.name: path for path in failed_root.iterdir()}
    _require(
        set(entries) == set(ALIAS_ORDER)
        and all(path.is_file() and not _linklike(path) for path in entries.values()),
        "preserved V3 output root is not exact immutable aliases2",
    )
    incident_aliases = incident["postfailure_state"]["aliases"]
    incident_keys = {
        "FIELD_CENSUS_LOCK.json": "FIELD_CENSUS_LOCK",
        "PROVENANCE_LEDGER.parquet": "PROVENANCE_LEDGER",
    }
    result: dict[str, Any] = {}
    for name, spec in V3_ALIAS_SPECS.items():
        source = path_under(root, spec["source"], label=f"preserved V3 source {name}")
        final = path_under(root, spec["final"], label=f"preserved V3 alias {name}")
        observed = file_identity(final, relative_to=root)
        _require(
            observed["size_bytes"] == spec["size_bytes"]
            and observed["sha256"] == spec["sha256"],
            f"preserved V3 alias identity mismatch: {name}",
        )
        _require(not os.path.samefile(source, final), f"preserved V3 alias shares source inode: {name}")
        if hasattr(source.stat(), "st_nlink"):
            _require(
                source.stat().st_nlink == 1 and final.stat().st_nlink == 1,
                f"preserved V3 alias link count mismatch: {name}",
            )
        _require_exact_tree(
            incident_aliases[incident_keys[name]]["alias_identity"],
            observed,
            label=f"incident/preserved V3 alias crosslink: {name}",
        )
        result[name] = observed
    return {
        "historical_controls": historical,
        "aliases": result,
        "v3_output_root_entries": list(ALIAS_ORDER),
    }


def validate_postrun_namespace_state(
    artifact_root: Path,
    staged_output_identities: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove aliases+seven stages and no final publication at audit time."""

    root = path_under(
        artifact_root,
        OUTPUT_ROOT_RELATIVE.rstrip("/"),
        label="V5 postrun output root",
    )
    transaction = path_under(
        artifact_root,
        TRANSACTION_ROOT_RELATIVE.rstrip("/"),
        label="V5 postrun transaction root",
    )
    _require(root.is_dir() and not _linklike(root), "V5 postrun output root absent/link-like")
    _require(
        transaction.is_dir() and not _linklike(transaction),
        "V5 postrun transaction root absent/link-like",
    )
    root_entries = {path.name: path for path in root.iterdir()}
    _require(
        set(root_entries) == {".transaction_v5", *ALIAS_ORDER},
        "V5 postrun root is not aliases2 plus transaction directory",
    )
    _require(root_entries[".transaction_v5"].is_dir(), "V5 transaction entry is not a directory")
    for alias in ALIAS_ORDER:
        _require(root_entries[alias].is_file() and not _linklike(root_entries[alias]), f"V3 alias missing/link-like: {alias}")
    transaction_entries = {path.name: path for path in transaction.iterdir()}
    _require(
        set(transaction_entries) == set(RUNNER_PUBLICATION_ORDER),
        "V5 postrun transaction is not exact staged7",
    )
    _require(
        all(path.is_file() and not _linklike(path) for path in transaction_entries.values()),
        "V5 postrun transaction contains a non-regular staged entry",
    )
    validate_alias_predata_closure(
        artifact_root, staged_output_identities["aliases"]
    )
    validate_final_alias_schema_closure(artifact_root, amendment)
    validate_staged_output_identities(
        staged_output_identities, artifact_root=artifact_root
    )
    return dict(AUDITOR_POSTSTATE)


def validate_authority_chain(
    artifact_root: Path,
    authorization_path: Path,
    independent_go_path: Path,
    postrun_path: Path,
    *,
    repo_root: Path = REPO,
    validate_staged_content: bool = True,
    actual_orig_argv: Any,
) -> dict[str, Any]:
    """Validate the documentary chain and audited prepublication filesystem.

    This routine performs JSON/CSV, byte-identity, and one final provenance-alias
    footer read.  It never opens either staged Parquet output or any Parquet
    row group through a logical-value reader.
    """

    root = Path(os.path.abspath(artifact_root))
    repo = Path(os.path.abspath(repo_root))
    amendment = validate_amendment(root)
    validate_sealer_runtime_identity(amendment)
    code_seal, code_seal_identity_record = load_control_create_if_absent_identity(
        root, CODE_SEAL_RELATIVE, label="V5 code seal"
    )
    authorization, authorization_identity_record = load_control_create_if_absent_identity(
        root,
        AUTHORIZATION_RELATIVE,
        label="V5 authorization",
        supplied_path=authorization_path,
    )
    review, review_identity_record = load_control_create_if_absent_identity(
        root, REVIEW_RELATIVE, label="V5 independent review"
    )
    independent_go, go_identity_record = load_control_create_if_absent_identity(
        root,
        GO_RELATIVE,
        label="V5 independent GO",
        supplied_path=independent_go_path,
    )
    postrun, postrun_identity_record = load_control_create_if_absent_identity(
        root,
        POSTRUN_RELATIVE,
        label="V5 independent postrun pass",
        supplied_path=postrun_path,
    )

    controls = {
        "code_seal": code_seal,
        "authorization": authorization,
        "review": review,
        "go": independent_go,
        "postrun": postrun,
    }
    for kind, payload in controls.items():
        validate_control_payload(
            kind,
            payload,
            amendment=amendment,
            # POSTRUN is validated documentary-only here.  Its staged/output
            # byte reads are deliberately deferred until after this process's
            # actual argv has been bound to the authorized sealer command.
            artifact_root=None if kind == "postrun" else root,
            repo_root=repo,
            rehash_code=True,
        )

    for observed, expected, label in (
        (authorization["code_seal"], code_seal_identity_record, "authorization code-seal crosslink"),
        (review["code_seal"], code_seal_identity_record, "review code-seal crosslink"),
        (review["authorization"], authorization_identity_record, "review authorization crosslink"),
        (independent_go["code_seal"], code_seal_identity_record, "GO code-seal crosslink"),
        (independent_go["authorization"], authorization_identity_record, "GO authorization crosslink"),
        (independent_go["independent_review"], review_identity_record, "GO review crosslink"),
        (postrun["code_seal"], code_seal_identity_record, "postrun code-seal crosslink"),
        (postrun["authorization"], authorization_identity_record, "postrun authorization crosslink"),
        (postrun["independent_review"], review_identity_record, "postrun review crosslink"),
        (postrun["independent_go"], go_identity_record, "postrun GO crosslink"),
    ):
        _require_exact_tree(observed, expected, label=label)

    attempt_id = authorization["attempt_id"]
    _require(
        all(control.get("attempt_id") == attempt_id for control in (review, independent_go, postrun)),
        "V5 control attempt_id chain mismatch",
    )
    code_identities = code_seal["code_identities"]
    _require(
        all(
            _exact_tree_equal(control.get("code_identities"), code_identities)
            for control in (authorization, review, independent_go, postrun)
        ),
        "V5 code identity chain mismatch",
    )

    chronology = [
        _utc_order_value(amendment["created_utc"], label="V3 amendment"),
        _utc_order_value(AMENDMENT_V4_CREATED_UTC, label="V4 amendment"),
        _utc_order_value(INCIDENT_CREATED_UTC, label="V3 incident"),
        _utc_order_value(AMENDMENT_V5_CREATED_UTC, label="V5 amendment"),
        _utc_order_value(code_seal["created_utc"], label="V5 code seal"),
        _utc_order_value(authorization["created_utc"], label="V5 authorization"),
        _utc_order_value(review["created_utc"], label="V5 independent review"),
        _utc_order_value(independent_go["created_utc"], label="V5 independent GO"),
        _utc_order_value(postrun["created_utc"], label="V5 independent postrun pass"),
    ]
    _require(
        all(left < right for left, right in zip(chronology, chronology[1:])),
        "V5 control chronology is not strictly increasing",
    )
    _require(
        _utc_order_value(code_seal["test_evidence"]["completed_utc"], label="V5 test evidence")
        <= chronology[4],
        "V5 test evidence completed after code seal",
    )
    _require(
        chronology[3]
        < _utc_order_value(
            code_seal["test_evidence"]["completed_utc"],
            label="V5 test evidence",
        ),
        "V5 test evidence completed before the V5 documentary amendment",
    )

    validate_actual_execution_argv(
        actual_orig_argv,
        authorization["required_commands"]["sealer"],
    )

    validate_historical_v3_failure_state(root)

    validate_postrun_namespace_state(
        root, postrun["staged_output_identities"], amendment
    )
    validated_threadpool: Mapping[str, Any] | None = None
    staged_decision: Mapping[str, Any] | None = None
    staged_fit_ledger: Mapping[str, Any] | None = None
    runner_stdout_payload: Mapping[str, Any] | None = None
    if validate_staged_content:
        validate_staged_csv_outputs(root, amendment)
        staged_json = validate_staged_json_outputs(root, amendment)
        decision = staged_json["decision"]
        staged_decision = decision
        staged_fit_ledger = staged_json["fit_ledger"]
        audit_decision = postrun["audit_result"]["stdout_payload"]["decision"]
        _require(
            audit_decision
            == {
                "status": decision["status"],
                "selected_family": decision["selected_family"],
                "global_ambiguity": decision["global_ambiguity"],
            },
            "V5 auditor decision does not match staged decision",
        )
        runner_payload = postrun["audit_result"]["runner_result"][
            "stdout_payload"
        ]
        runner_stdout_payload = runner_payload
        validate_runner_stdout(
            runner_payload,
            attempt_id=attempt_id,
            amendment=amendment,
            staged_output_identities=postrun["staged_output_identities"],
            expected_decision=audit_decision,
            fit_ledger=staged_json["fit_ledger"],
        )
        validated_threadpool = dict(
            runner_payload["threadpool_info_before_inside_after"]
        )
    static = static_self_validate()
    _require(
        static["models_fit"] == 0
        and static["predictions_computed"] == 0
        and static["parquet_logical_value_reads"] == 0
        and postrun["audit_result"]["stdout_payload"]["models_fit"] == 0,
        "V5 no-fit sealer/auditor evidence mismatch",
    )
    return {
        "amendment": amendment,
        "controls": controls,
        "control_identities": {
            "code_seal": code_seal_identity_record,
            "authorization": authorization_identity_record,
            "independent_review": review_identity_record,
            "independent_go": go_identity_record,
            "postrun_pass": postrun_identity_record,
        },
        "attempt_id": attempt_id,
        "code_identities": code_identities,
        "staged_output_identities": postrun["staged_output_identities"],
        "staged_decision": staged_decision,
        "staged_fit_ledger": staged_fit_ledger,
        "runner_stdout_payload": runner_stdout_payload,
        "threadpool_info_before_inside_after": validated_threadpool,
        "models_fit_by_sealer": 0,
        "parquet_logical_value_reads_by_sealer": 0,
    }


def build_code_seal_payload(
    *,
    amendment: Mapping[str, Any],
    code_identities: Mapping[str, Any],
    test_evidence: Mapping[str, Any],
    created_utc: str,
    repo_root: Path = REPO,
) -> dict[str, Any]:
    amendment_created = _utc_order_value(
        amendment.get("created_utc"), label="V3 amendment"
    )
    amendment_v4_created = _utc_order_value(
        AMENDMENT_V4_CREATED_UTC, label="V4 amendment"
    )
    incident_created = _utc_order_value(
        INCIDENT_CREATED_UTC, label="V3 failure incident"
    )
    amendment_v5_created = _utc_order_value(
        AMENDMENT_V5_CREATED_UTC, label="V5 amendment"
    )
    evidence_completed = _utc_order_value(
        test_evidence.get("completed_utc"), label="V5 test evidence"
    )
    seal_created = _utc_order_value(created_utc, label="V5 code seal")
    _require(
        amendment_created
        < amendment_v4_created
        < incident_created
        < amendment_v5_created
        < evidence_completed
        <= seal_created,
        "V5 code-seal chronology mismatch",
    )
    payload = {
        "schema_version": 5,
        "artifact_type": CONTROL_SPECS["code_seal"]["artifact_type"],
        "status": CONTROL_SPECS["code_seal"]["status"],
        "created_utc": created_utc,
        "amendment": amendment_identity(),
        "amendment_v4": amendment_v4_identity(),
        "amendment_v5": amendment_v5_identity(),
        "code_identities": dict(code_identities),
        "test_evidence": dict(test_evidence),
        "output_namespace": dict(OUTPUT_NAMESPACE),
        "authority_scope": dict(DOCUMENTARY_SCOPE),
        "required_next_controls": dict(REQUIRED_NEXT_CONTROLS),
        "publisher": CONTROL_SPECS["code_seal"]["publisher"],
    }
    validate_control_payload(
        "code_seal",
        payload,
        amendment=amendment,
        repo_root=repo_root,
    )
    return payload


def load_bound_parent_plan(
    artifact_root: Path, amendment: Mapping[str, Any]
) -> dict[str, Any]:
    record = amendment.get("bound_inputs", {}).get("parent_plan")
    _require(
        isinstance(record, Mapping) and set(record) == FILE_IDENTITY_KEYS,
        "bound parent-plan identity schema mismatch",
    )
    parent_path = path_under(
        artifact_root, str(record["path"]), label="bound V3 parent plan"
    )
    validate_identity_record(
        record,
        parent_path,
        label="bound V3 parent plan",
        relative_to=artifact_root,
        rehash=True,
    )
    parent = strict_json_load(parent_path, label="bound V3 parent plan")
    spatial_method = parent.get("decode_contract", {}).get("spatial_method")
    _require(
        spatial_method
        == "deterministic bilinear interpolation on regular_ll grid, then capacity-weighted group mean",
        "bound parent-plan spatial method mismatch",
    )
    return parent


def _code_config_runtime_identities(
    amendment: Mapping[str, Any], code_identities: Mapping[str, Any]
) -> dict[str, Any]:
    _require(
        isinstance(code_identities, Mapping)
        and set(code_identities) == set(CODE_ROLE_PATHS),
        "final code identity role set mismatch",
    )
    for role, record in code_identities.items():
        _require(
            isinstance(record, Mapping) and set(record) == FILE_IDENTITY_KEYS,
            f"final code identity schema mismatch: {role}",
        )
    return {
        "amendment": amendment_identity(),
        "amendment_v4": amendment_v4_identity(),
        "amendment_v5": amendment_v5_identity(),
        "code_identities": dict(code_identities),
        "runtime_identity": amendment["runtime_identity"],
    }


def _family_lock_unit(column: str) -> str:
    if column == "HPBL_surface":
        return "m"
    if column in {"ENDPOINT_DIRECTION_COS", "ENDPOINT_DIRECTION_SIN"}:
        return "dimensionless"
    return "m/s"


def _family_lock_grib_selector(column: str) -> dict[str, str]:
    if column == "HPBL_surface":
        return {"variable": "HPBL", "level": "surface"}
    match = re.fullmatch(r"(UGRD|VGRD)_([0-9]+)mb", column)
    _require(match is not None, f"family-lock raw GRIB selector is unknown: {column}")
    assert match is not None
    _require(
        int(match.group(2)) in {925, 950, 975, 1000},
        f"family-lock pressure level is unfrozen: {column}",
    )
    return {
        "variable": match.group(1),
        "level": f"{match.group(2)} mb",
    }


def _family_lock_terminal_key(
    decision: Mapping[str, Any], amendment: Mapping[str, Any]
) -> str:
    statuses = amendment["output_contract"]["decision_status_literals_exact"]
    matches = [key for key, value in statuses.items() if value == decision.get("status")]
    _require(len(matches) == 1, "family-lock decision terminal status mismatch")
    return matches[0]


def _expected_family_lock_payload(
    *,
    amendment: Mapping[str, Any],
    decision: Mapping[str, Any],
    code_identities: Mapping[str, Any],
    postrun_pass_identity: Mapping[str, Any],
    parent_plan: Mapping[str, Any],
) -> dict[str, Any]:
    terminal_key = _family_lock_terminal_key(decision, amendment)
    positive = terminal_key == "positive"
    selected_raw = decision["selected_raw_columns"] if positive else None
    selected_derived = decision["selected_derived_columns"] if positive else None
    selected_columns = (
        [*selected_raw, *selected_derived]
        if positive
        else []
    )
    units = (
        {column: _family_lock_unit(column) for column in selected_columns}
        if positive
        else None
    )
    selectors = (
        {
            column: _family_lock_grib_selector(column)
            for column in selected_raw
        }
        if positive
        else None
    )
    input_contract = amendment["input_and_join_contract"]
    site_group_scope = {
        key: input_contract[key] for key in SITE_GROUP_SCOPE_KEYS
    }
    spatial_transforms = {
        "spatial_method": parent_plan["decode_contract"]["spatial_method"],
        "group_construction": amendment["derived_wind_contract"][
            "group_construction"
        ],
    }
    return {
        "schema_version": 5,
        "artifact_type": LOCK_ARTIFACT_TYPE,
        "status": LOCK_STATUS_BY_DECISION_STATUS[decision["status"]],
        "selected_family": decision["selected_family"] if positive else None,
        "ordered_raw_columns": selected_raw,
        "ordered_derived_columns": selected_derived,
        "units": units,
        "grib_selectors": selectors,
        "site_group_scope": site_group_scope,
        "spatial_transforms": spatial_transforms,
        "parent_and_independent_gate_results": decision,
        "bound_input_identities": amendment["bound_inputs"],
        "code_config_runtime_identities": _code_config_runtime_identities(
            amendment, code_identities
        ),
        "postrun_pass_identity": dict(postrun_pass_identity),
        "no_label_attestation": dict(DECISION_FORBIDDEN_ACCESS_ATTESTATION),
        "downstream_authority": amendment["output_contract"][
            "output_record_schemas_exact"
        ]["TARGET_FREE_FAMILY_LOCK_json"]["downstream_authority_literal"],
    }


def validate_family_lock_payload(
    payload: Any,
    *,
    amendment: Mapping[str, Any],
    decision: Mapping[str, Any],
    code_identities: Mapping[str, Any],
    postrun_pass_identity: Mapping[str, Any],
    parent_plan: Mapping[str, Any],
) -> dict[str, Any]:
    validate_decision_payload(decision, amendment)
    lock_schema = amendment["output_contract"]["output_record_schemas_exact"][
        "TARGET_FREE_FAMILY_LOCK_json"
    ]
    _require(
        set(lock_schema["top_keys_exact"]) == LOCK_TOP_KEYS
        and len(lock_schema["top_keys_exact"]) == len(LOCK_TOP_KEYS),
        "amendment family-lock top schema differs",
    )
    _require(
        isinstance(postrun_pass_identity, Mapping)
        and set(postrun_pass_identity) == FILE_IDENTITY_KEYS
        and postrun_pass_identity.get("path") == POSTRUN_RELATIVE,
        "family-lock postrun identity mismatch",
    )
    _require(
        isinstance(payload, Mapping) and set(payload) == LOCK_TOP_KEYS,
        "family-lock top schema mismatch",
    )
    expected = _expected_family_lock_payload(
        amendment=amendment,
        decision=decision,
        code_identities=code_identities,
        postrun_pass_identity=postrun_pass_identity,
        parent_plan=parent_plan,
    )
    _require_exact_tree(payload, expected, label="family-lock exact payload")
    negative_nulls = lock_schema["negative_tombstone_null_fields"]
    terminal_key = _family_lock_terminal_key(decision, amendment)
    if terminal_key != "positive":
        _require(
            all(payload[field] is None for field in negative_nulls),
            "family-lock negative tombstone nullability mismatch",
        )
    else:
        expected_columns = [
            *payload["ordered_raw_columns"],
            *payload["ordered_derived_columns"],
        ]
        _require(
            set(payload["units"]) == set(expected_columns)
            and set(payload["grib_selectors"])
            == set(payload["ordered_raw_columns"]),
            "family-lock positive unit/selector coverage mismatch",
        )
        _require(
            all(
                isinstance(selector, Mapping)
                and set(selector) == set(GRIB_SELECTOR_KEYS)
                for selector in payload["grib_selectors"].values()
            ),
            "family-lock GRIB selector schema mismatch",
        )
    return dict(payload)


def build_family_lock_payload(
    *,
    amendment: Mapping[str, Any],
    decision: Mapping[str, Any],
    code_identities: Mapping[str, Any],
    postrun_pass_identity: Mapping[str, Any],
    parent_plan: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _expected_family_lock_payload(
        amendment=amendment,
        decision=decision,
        code_identities=code_identities,
        postrun_pass_identity=postrun_pass_identity,
        parent_plan=parent_plan,
    )
    return validate_family_lock_payload(
        payload,
        amendment=amendment,
        decision=decision,
        code_identities=code_identities,
        postrun_pass_identity=postrun_pass_identity,
        parent_plan=parent_plan,
    )


def _finalized_output_identity(
    name: str, record: Mapping[str, Any]
) -> dict[str, Any]:
    _require(
        isinstance(record, Mapping) and set(record) == OUTPUT_IDENTITY_KEYS,
        f"manifest source output identity schema mismatch: {name}",
    )
    result = dict(record)
    result["path"] = FINAL_RELATIVES[name]
    return result


def _manifest_oof_slot_counts(
    *,
    amendment: Mapping[str, Any],
    decision: Mapping[str, Any],
    fit_ledger: Mapping[str, Any],
    staged_output_identities: Mapping[str, Any],
) -> dict[str, int]:
    oof_identity = staged_output_identities["runner_staged_outputs"][
        "TARGET_FREE_DUPLICATE_OOF_V5.parquet"
    ]
    planned = oof_identity["row_count"]
    site_rows = amendment["input_and_join_contract"]["external_site_rows"]
    _require(
        site_rows == 19_584 and site_rows % len(FOLD_ENUM) == 0,
        "manifest OOF site/fold accounting source mismatch",
    )
    heldout_rows = site_rows // len(FOLD_ENUM)
    executed = fit_ledger["executed_counts"]
    skipped = fit_ledger["skipped_counts"]
    completed_slots = executed["completed_primary_model_units"] * heldout_rows
    skipped_slots = skipped["skipped_primary_model_units"] * heldout_rows
    selected_records = [
        record
        for record in decision["raw_component_results"]
        if record["independent_estimator"] is not None
    ]
    selected_slots = len(selected_records) * site_rows
    result = {
        "planned_oof_slots": planned,
        "completed_oof_slots": completed_slots,
        "skipped_oof_slots": skipped_slots,
        "independent_selected_oof_slots": selected_slots,
    }
    _require(
        all(type(value) is int and value >= 0 for value in result.values())
        and completed_slots + skipped_slots == planned
        and completed_slots % heldout_rows == 0
        and skipped_slots % heldout_rows == 0
        and selected_slots % site_rows == 0
        and selected_slots <= completed_slots
        and selected_slots <= len(decision["raw_component_results"]) * site_rows,
        "manifest OOF slot accounting mismatch",
    )
    primary_slots = fit_ledger["unit_slots"][
        : FIT_LEDGER_PLANNED_COUNTS["primary_model_units"]
    ]
    for record in selected_records:
        response = record["response"]
        estimator = record["independent_estimator"]
        selected_fold_slots = [
            slot
            for slot in primary_slots
            if slot.get("response") == response
            and slot.get("estimator_unit") == estimator
        ]
        _require(
            len(selected_fold_slots) == len(FOLD_ENUM)
            and {slot.get("fold") for slot in selected_fold_slots}
            == set(FOLD_ENUM)
            and all(
                slot.get("unit_status") == "COMPLETED"
                and slot.get("fit_completed") is True
                for slot in selected_fold_slots
            ),
            "manifest selected OOF ledger crosslink mismatch: "
            f"{response}/{estimator}",
        )
    return result


def _expected_run_manifest_payload(
    *,
    amendment: Mapping[str, Any],
    decision: Mapping[str, Any],
    fit_ledger: Mapping[str, Any],
    code_identities: Mapping[str, Any],
    control_identities: Mapping[str, Any],
    controls: Mapping[str, Any],
    staged_output_identities: Mapping[str, Any],
    lock_output_identity: Mapping[str, Any],
    threadpool_info: Mapping[str, Any],
    no_forbidden_access_attestation: Mapping[str, Any],
    created_utc: str,
) -> dict[str, Any]:
    runner = staged_output_identities["runner_staged_outputs"]
    outputs = {
        name: _finalized_output_identity(name, runner[name])
        for name in RUNNER_PUBLICATION_ORDER
    }
    outputs["TARGET_FREE_FAMILY_LOCK.json"] = _finalized_output_identity(
        "TARGET_FREE_FAMILY_LOCK.json", lock_output_identity
    )
    fit_counts = {
        key: fit_ledger[key] for key in MANIFEST_FIT_COUNT_KEYS
    }
    return {
        "schema_version": 5,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "status": MANIFEST_STATUS,
        "terminal_state": decision["status"],
        "created_utc": created_utc,
        "code_config_runtime_identities": _code_config_runtime_identities(
            amendment, code_identities
        ),
        "control_identities": {
            key: control_identities[key]
            for key in MANIFEST_CONTROL_IDENTITY_KEYS
        },
        "alias_identities": {
            name: staged_output_identities["aliases"][name]
            for name in ALIAS_ORDER
        },
        "output_identities": outputs,
        "planned_completed_skipped_fit_counts": fit_counts,
        "oof_slot_counts": _manifest_oof_slot_counts(
            amendment=amendment,
            decision=decision,
            fit_ledger=fit_ledger,
            staged_output_identities=staged_output_identities,
        ),
        "threadpool_info_before_inside_after": threadpool_info,
        "no_forbidden_access_attestation": no_forbidden_access_attestation,
        "manifest_is_last_commit_marker": True,
    }


def validate_run_manifest_payload(
    payload: Any,
    *,
    amendment: Mapping[str, Any],
    decision: Mapping[str, Any],
    fit_ledger: Mapping[str, Any],
    code_identities: Mapping[str, Any],
    control_identities: Mapping[str, Any],
    controls: Mapping[str, Any],
    staged_output_identities: Mapping[str, Any],
    lock_output_identity: Mapping[str, Any],
    threadpool_info: Mapping[str, Any],
    no_forbidden_access_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    validate_decision_payload(decision, amendment)
    validate_fit_ledger_payload(
        fit_ledger, amendment, expected_status=str(decision["status"])
    )
    validate_staged_output_identities(staged_output_identities)
    validate_threadpool_evidence_structure(threadpool_info, amendment)
    _require_exact_tree(
        no_forbidden_access_attestation,
        DECISION_FORBIDDEN_ACCESS_ATTESTATION,
        label="manifest forbidden-access attestation",
    )
    manifest_schema = amendment["output_contract"]["output_record_schemas_exact"][
        "TRACK_A_TARGET_FREE_RUN_MANIFEST_json"
    ]
    _require(
        set(manifest_schema["top_keys_exact"]) == MANIFEST_TOP_KEYS
        and len(manifest_schema["top_keys_exact"]) == len(MANIFEST_TOP_KEYS)
        and manifest_schema["publish_last"] is True,
        "amendment run-manifest schema differs",
    )
    _require(
        isinstance(payload, Mapping) and set(payload) == MANIFEST_TOP_KEYS,
        "run-manifest top schema mismatch",
    )
    created_utc = require_created_utc(
        payload.get("created_utc"), label="run manifest"
    )
    _require(
        _utc_order_value(AMENDMENT_V5_CREATED_UTC, label="V5 amendment")
        < _utc_order_value(created_utc, label="run manifest"),
        "run-manifest chronology does not follow the V5 documentary amendment",
    )
    _require(
        isinstance(controls, Mapping)
        and set(controls) == {"code_seal", "authorization", "review", "go", "postrun"},
        "run-manifest control document set mismatch",
    )
    _require(
        _utc_order_value(controls["postrun"]["created_utc"], label="postrun")
        < _utc_order_value(created_utc, label="run manifest"),
        "run-manifest chronology does not follow postrun pass",
    )
    _require(
        isinstance(control_identities, Mapping)
        and len(control_identities) == len(MANIFEST_CONTROL_IDENTITY_KEYS)
        and set(control_identities) == set(MANIFEST_CONTROL_IDENTITY_KEYS),
        "run-manifest control identity set mismatch",
    )
    for name, record in control_identities.items():
        _require(
            isinstance(record, Mapping) and set(record) == FILE_IDENTITY_KEYS,
            f"run-manifest control identity schema mismatch: {name}",
        )
    _require(
        isinstance(lock_output_identity, Mapping)
        and set(lock_output_identity) == OUTPUT_IDENTITY_KEYS
        and lock_output_identity.get("path")
        == SEALER_STAGED_RELATIVES["TARGET_FREE_FAMILY_LOCK.json"]
        and lock_output_identity.get("format") == "JSON"
        and lock_output_identity.get("row_count") == 1
        and lock_output_identity.get("logical_sha256")
        == lock_output_identity.get("sha256"),
        "run-manifest lock output identity mismatch",
    )
    expected = _expected_run_manifest_payload(
        amendment=amendment,
        decision=decision,
        fit_ledger=fit_ledger,
        code_identities=code_identities,
        control_identities=control_identities,
        controls=controls,
        staged_output_identities=staged_output_identities,
        lock_output_identity=lock_output_identity,
        threadpool_info=threadpool_info,
        no_forbidden_access_attestation=no_forbidden_access_attestation,
        created_utc=created_utc,
    )
    _require_exact_tree(payload, expected, label="run-manifest exact payload")
    _require(
        set(payload["output_identities"]) == set(MANIFEST_OUTPUT_ORDER)
        and "TRACK_A_TARGET_FREE_RUN_MANIFEST.json"
        not in payload["output_identities"],
        "run-manifest output8/self-identity closure mismatch",
    )
    _require(
        set(payload["planned_completed_skipped_fit_counts"])
        == set(MANIFEST_FIT_COUNT_KEYS)
        and set(payload["oof_slot_counts"]) == set(MANIFEST_OOF_COUNT_KEYS),
        "run-manifest count-map schema mismatch",
    )
    return dict(payload)


def build_run_manifest_payload(
    *,
    amendment: Mapping[str, Any],
    decision: Mapping[str, Any],
    fit_ledger: Mapping[str, Any],
    code_identities: Mapping[str, Any],
    control_identities: Mapping[str, Any],
    controls: Mapping[str, Any],
    staged_output_identities: Mapping[str, Any],
    lock_output_identity: Mapping[str, Any],
    threadpool_info: Mapping[str, Any],
    no_forbidden_access_attestation: Mapping[str, Any],
    created_utc: str,
) -> dict[str, Any]:
    payload = _expected_run_manifest_payload(
        amendment=amendment,
        decision=decision,
        fit_ledger=fit_ledger,
        code_identities=code_identities,
        control_identities=control_identities,
        controls=controls,
        staged_output_identities=staged_output_identities,
        lock_output_identity=lock_output_identity,
        threadpool_info=threadpool_info,
        no_forbidden_access_attestation=no_forbidden_access_attestation,
        created_utc=created_utc,
    )
    return validate_run_manifest_payload(
        payload,
        amendment=amendment,
        decision=decision,
        fit_ledger=fit_ledger,
        code_identities=code_identities,
        control_identities=control_identities,
        controls=controls,
        staged_output_identities=staged_output_identities,
        lock_output_identity=lock_output_identity,
        threadpool_info=threadpool_info,
        no_forbidden_access_attestation=no_forbidden_access_attestation,
    )


def stage_sealer_json_payload(
    artifact_root: Path, name: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    _require(name in SEALER_STAGED_RELATIVES, f"unknown sealer JSON stage: {name}")
    relative = SEALER_STAGED_RELATIVES[name]
    stage = path_under(artifact_root, relative, label=f"sealer JSON stage {name}")
    data = pretty_json_bytes(payload)
    stage_bytes_exclusive(stage, data)
    observed = file_identity(stage, relative_to=artifact_root)
    _require(
        observed["size_bytes"] == len(data)
        and observed["sha256"] == hashlib.sha256(data).hexdigest(),
        f"sealer JSON stage identity mismatch: {name}",
    )
    round_trip = strict_json_load(stage, label=f"sealer JSON stage {name}")
    _require_exact_tree(round_trip, payload, label=f"sealer JSON stage {name}")
    _require(stage.read_bytes() == pretty_json_bytes(round_trip), f"sealer JSON stage is not deterministic: {name}")
    return {
        **observed,
        "format": "JSON",
        "row_count": 1,
        "logical_sha256": observed["sha256"],
    }


def stage_bytes_exclusive(path: Path, data: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not _lexists(path), f"staging path already exists: {path}")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return file_identity(path)


def stage_copy_exclusive(
    source: Path,
    stage: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, Any]:
    _require(source.is_file() and not _linklike(source), f"alias source absent/link-like: {source}")
    _require(source.stat().st_size == expected_size and sha256_file(source) == expected_sha256, "alias source identity mismatch")
    stage.parent.mkdir(parents=True, exist_ok=True)
    _require(not _lexists(stage), f"alias stage already exists: {stage}")
    with source.open("rb") as reader, stage.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 << 20)
        writer.flush()
        os.fsync(writer.fileno())
    record = file_identity(stage)
    _require(record["size_bytes"] == expected_size and record["sha256"] == expected_sha256, "alias staged copy identity mismatch")
    _require(not os.path.samefile(source, stage), "alias staged copy unexpectedly shares source inode")
    return record


def publish_staged_create_if_absent(
    staged: Path,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Sole final-name publisher: hard-link create-if-absent, then unlink stage."""

    _require(staged.is_file() and not _linklike(staged), f"staged file absent/link-like: {staged}")
    _require(not _lexists(destination), f"final destination already exists: {destination}")
    _require(staged.stat().st_size == expected_size and sha256_file(staged) == expected_sha256, f"staged identity mismatch: {staged}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    linked = False
    try:
        os.link(staged, destination, follow_symlinks=False)
        linked = True
        _require(os.path.samefile(staged, destination), "published destination is not staged hard link")
        with destination.open("r+b") as stream:
            os.fsync(stream.fileno())
        staged.unlink()
        observed = file_identity(destination)
        _require(observed["size_bytes"] == expected_size and observed["sha256"] == expected_sha256, "published destination failed rehash")
        return observed
    except Exception as exc:
        if linked and _lexists(destination):
            try:
                observed = file_identity(destination)
            except Exception:
                observed = {"path": str(destination), "size_bytes": expected_size, "sha256": expected_sha256}
            raise PostPublicationError(
                "destination was published but a later verification/unlink step failed",
                observed,
            ) from exc
        if isinstance(exc, TargetFreeSealError):
            raise
        raise TargetFreeSealError(f"create-if-absent publication failed: {destination}") from exc


def publish_alias_pair_predata(
    artifact_root: Path,
    alias_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Publish both byte aliases before any logical value-access route is called."""

    _require(tuple(alias_specs) == ALIAS_ORDER, "alias publication order mismatch")
    resolved: dict[str, tuple[Path, Path, Path, Mapping[str, Any]]] = {}
    for name in ALIAS_ORDER:
        spec = alias_specs[name]
        source = path_under(artifact_root, str(spec["source"]), label=f"alias source {name}")
        stage = path_under(artifact_root, str(spec["stage"]), label=f"alias stage {name}")
        final = path_under(artifact_root, str(spec["final"]), label=f"alias final {name}")
        _require(not _lexists(stage) and not _lexists(final), f"alias prestate not absent: {name}")
        resolved[name] = (source, stage, final, spec)
    for name in ALIAS_ORDER:
        source, stage, _final, spec = resolved[name]
        stage_copy_exclusive(source, stage, expected_size=int(spec["size_bytes"]), expected_sha256=str(spec["sha256"]))
    published: list[str] = []
    result: dict[str, dict[str, Any]] = {}
    try:
        for name in ALIAS_ORDER:
            source, stage, final, spec = resolved[name]
            observed = publish_staged_create_if_absent(
                stage,
                final,
                expected_size=int(spec["size_bytes"]),
                expected_sha256=str(spec["sha256"]),
            )
            _require(not os.path.samefile(source, final), f"persistent source hardlink forbidden: {name}")
            if hasattr(source.stat(), "st_nlink"):
                _require(source.stat().st_nlink == 1 and final.stat().st_nlink == 1, f"alias persistent link count mismatch: {name}")
            observed["path"] = str(spec["final"])
            result[name] = observed
            published.append(name)
    except Exception as exc:
        physically_published = [
            name for name in ALIAS_ORDER if _lexists(resolved[name][2])
        ]
        if published or physically_published:
            raise PartialPublicationError(
                "partial V5 alias publication is terminal; append-only V6 supersession required",
                physically_published or published,
            ) from exc
        raise
    return result


def publish_normative_aliases_predata(
    artifact_root: Path, amendment: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Validate semantic bundles, then execute the frozen two-alias protocol."""

    bound = amendment["bound_inputs"]
    by_path = {
        str(record["path"]): record
        for record in bound.values()
        if isinstance(record, Mapping) and FILE_IDENTITY_KEYS.issubset(record)
    }
    for name, spec in ALIAS_SPECS.items():
        for relative in spec["bundle"]:
            _require(relative in by_path, f"alias bundle identity absent from V3: {name}/{relative}")
            record = by_path[relative]
            expected = path_under(artifact_root, relative, label=f"alias bundle {relative}")
            validate_identity_record(
                {key: record[key] for key in FILE_IDENTITY_KEYS},
                expected,
                label=f"alias bundle {relative}",
                relative_to=artifact_root,
                rehash=True,
            )
    return publish_alias_pair_predata(artifact_root, ALIAS_SPECS)


def validate_alias_predata_closure(
    artifact_root: Path,
    aliases: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    _require(set(aliases) == set(ALIAS_ORDER), "alias closure identity set mismatch")
    result: dict[str, dict[str, Any]] = {}
    for name, spec in ALIAS_SPECS.items():
        source = path_under(artifact_root, spec["source"], label=f"alias source {name}")
        final = path_under(artifact_root, spec["final"], label=f"alias final {name}")
        record = validate_identity_record(
            aliases[name], final, label=f"alias final {name}", relative_to=artifact_root, rehash=True
        )
        _require(record["size_bytes"] == spec["size_bytes"] and record["sha256"] == spec["sha256"], f"alias final frozen identity mismatch: {name}")
        _require(not os.path.samefile(source, final), f"alias final shares source inode: {name}")
        if hasattr(source.stat(), "st_nlink"):
            _require(source.stat().st_nlink == 1 and final.stat().st_nlink == 1, f"alias closure link count mismatch: {name}")
        result[name] = record
    return result


def _validate_census_alias_identity(
    artifact_root: Path,
    record: Any,
    *,
    label: str,
) -> str:
    _require(
        isinstance(record, Mapping) and set(record) == FILE_IDENTITY_KEYS,
        f"{label} identity schema mismatch",
    )
    raw_path = record.get("path")
    size = record.get("size_bytes")
    digest = record.get("sha256")
    _require(isinstance(raw_path, str) and bool(raw_path), f"{label} path invalid")
    _require(
        isinstance(size, int) and not isinstance(size, bool) and size >= 0,
        f"{label} size invalid",
    )
    _require(
        isinstance(digest, str) and LOWER_SHA256.fullmatch(digest) is not None,
        f"{label} SHA invalid",
    )
    raw_path_object = Path(raw_path)
    if raw_path_object.is_absolute():
        expected = Path(os.path.abspath(raw_path_object))
        _require(
            normalized_absolute(expected) == raw_path.replace("\\", "/"),
            f"{label} absolute path normalization mismatch",
        )
        _require_no_link_chain(expected, REPO, label=label)
    else:
        _require("\\" not in raw_path, f"{label} relative path has backslash")
        expected = path_under(artifact_root, raw_path, label=label)
    observed = file_identity(expected)
    _require(
        observed["size_bytes"] == size and observed["sha256"] == digest,
        f"{label} referenced identity mismatch",
    )
    return raw_path


def validate_census_alias_semantics(
    artifact_root: Path,
    payload: Any,
) -> dict[str, Any]:
    _require(
        isinstance(payload, Mapping) and set(payload) == CENSUS_ALIAS_TOP_KEYS,
        "FIELD_CENSUS_LOCK semantic top schema mismatch",
    )
    _require(
        isinstance(payload.get("schema_version"), int)
        and not isinstance(payload["schema_version"], bool)
        and payload["schema_version"] == 1,
        "FIELD_CENSUS_LOCK semantic schema_version mismatch",
    )
    _require(
        payload.get("artifact_type") == "NOAA_GFS_MULTISEASON_CENSUS_MANIFEST",
        "FIELD_CENSUS_LOCK semantic artifact_type mismatch",
    )
    require_created_utc(payload.get("created_utc"), label="FIELD_CENSUS_LOCK")
    _require(payload.get("labels_read") is False, "FIELD_CENSUS_LOCK labels_read mismatch")
    _require(payload.get("submission_csv_created") is False, "FIELD_CENSUS_LOCK submission flag mismatch")
    for field in ("models_fit", "raw_downloaded_bytes", "raw_network_requests"):
        _require(
            isinstance(payload.get(field), int)
            and not isinstance(payload[field], bool)
            and payload[field] == 0,
            f"FIELD_CENSUS_LOCK {field} mismatch",
        )
    for field in CENSUS_ALIAS_IDENTITY_FIELDS:
        _validate_census_alias_identity(
            artifact_root, payload[field], label=f"FIELD_CENSUS_LOCK.{field}"
        )
    checkpoints = payload.get("progress_checkpoints")
    _require(
        isinstance(checkpoints, list) and len(checkpoints) == 13,
        "FIELD_CENSUS_LOCK progress checkpoint count mismatch",
    )
    checkpoint_paths = [
        _validate_census_alias_identity(
            artifact_root,
            record,
            label=f"FIELD_CENSUS_LOCK.progress_checkpoints[{index}]",
        )
        for index, record in enumerate(checkpoints)
    ]
    _require(
        len(checkpoint_paths) == len(set(checkpoint_paths)),
        "FIELD_CENSUS_LOCK progress checkpoint paths are not unique",
    )
    return dict(payload)


def validate_final_alias_schema_closure(
    artifact_root: Path,
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    """Recheck final aliases without reading a Parquet logical value."""

    field_spec = ALIAS_SPECS["FIELD_CENSUS_LOCK.json"]
    field_path = path_under(
        artifact_root, field_spec["final"], label="FIELD_CENSUS_LOCK alias"
    )
    field_before = file_identity(field_path, relative_to=artifact_root)
    _require(
        field_before["size_bytes"] == field_spec["size_bytes"]
        and field_before["sha256"] == field_spec["sha256"],
        "FIELD_CENSUS_LOCK alias identity mismatch before JSON validation",
    )
    field_payload = strict_json_load(field_path, label="FIELD_CENSUS_LOCK alias")
    validate_census_alias_semantics(artifact_root, field_payload)
    field_after = file_identity(field_path, relative_to=artifact_root)
    _require_exact_tree(
        field_after, field_before,
        label="FIELD_CENSUS_LOCK alias identity across JSON validation",
    )

    provenance_spec = ALIAS_SPECS["PROVENANCE_LEDGER.parquet"]
    provenance_path = path_under(
        artifact_root,
        provenance_spec["final"],
        label="PROVENANCE_LEDGER alias",
    )
    provenance_before = file_identity(
        provenance_path, relative_to=artifact_root
    )
    _require(
        provenance_before["size_bytes"] == provenance_spec["size_bytes"]
        and provenance_before["sha256"] == provenance_spec["sha256"],
        "PROVENANCE_LEDGER alias identity mismatch before footer validation",
    )
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        _require(
            pa.__version__
            == amendment["runtime_identity"]["packages"]["pyarrow"],
            "PROVENANCE footer PyArrow version drift",
        )
        metadata = pq.read_metadata(provenance_path)
        observed_rows = int(metadata.num_rows)
        observed_names = tuple(metadata.schema.names)
    except Exception as exc:
        raise TargetFreeSealError(
            "PROVENANCE_LEDGER alias footer metadata is unreadable"
        ) from exc
    _require(
        len(observed_names) == len(set(observed_names)),
        "PROVENANCE_LEDGER alias footer has duplicate field names",
    )
    frozen = amendment["output_contract"]["upstream_alias_closure"][
        "aliases"
    ]["PROVENANCE_LEDGER.parquet"]
    _require(
        observed_rows == frozen["rows"] == 10_368,
        "PROVENANCE_LEDGER alias footer row count mismatch",
    )
    required = tuple(frozen["required_preregister_provenance_fields_exact"])
    missing = [name for name in required if name not in observed_names]
    _require(
        not missing,
        f"PROVENANCE_LEDGER alias footer lacks required exact15 fields: {missing}",
    )
    provenance_after = file_identity(
        provenance_path, relative_to=artifact_root
    )
    _require_exact_tree(
        provenance_after, provenance_before,
        label="PROVENANCE_LEDGER alias identity across footer validation",
    )
    return {
        "field_census_lock_identity": field_after,
        "provenance_ledger_identity": provenance_after,
        "provenance_footer_rows": observed_rows,
        "provenance_footer_field_names": list(observed_names),
        "required_exact15_present": True,
        "parquet_logical_values_read": 0,
    }


def publish_final_sequence(
    artifact_root: Path,
    staged_identities: Mapping[str, Mapping[str, Any]],
    *,
    order: Sequence[str] = FINAL_PUBLICATION_ORDER,
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    _require(tuple(order) == FINAL_PUBLICATION_ORDER, "final publication order mismatch")
    _require(order[-1] == "TRACK_A_TARGET_FREE_RUN_MANIFEST.json", "manifest is not last")
    _require(set(staged_identities) == set(FINAL_PUBLICATION_ORDER), "final staged identity set mismatch")
    resolved: dict[str, tuple[Path, Path, Mapping[str, Any]]] = {}
    for name in order:
        stage_relative = RUNNER_STAGED_RELATIVES.get(name, SEALER_STAGED_RELATIVES.get(name))
        _require(stage_relative is not None, f"unknown staged output: {name}")
        stage = path_under(artifact_root, stage_relative, label=f"final stage {name}")
        final = path_under(artifact_root, FINAL_RELATIVES[name], label=f"final output {name}")
        record = staged_identities[name]
        _require(isinstance(record, Mapping), f"staged identity absent: {name}")
        _require(record.get("path") == stage_relative, f"staged path mismatch: {name}")
        _require(isinstance(record.get("size_bytes"), int) and not isinstance(record["size_bytes"], bool) and int(record["size_bytes"]) > 0, f"staged size invalid: {name}")
        _require(isinstance(record.get("sha256"), str) and LOWER_SHA256.fullmatch(str(record["sha256"])) is not None, f"staged SHA invalid: {name}")
        _require(stage.is_file() and not _linklike(stage), f"staged output absent/link-like: {name}")
        _require(not _lexists(final), f"final output already exists: {name}")
        _require(stage.stat().st_size == record["size_bytes"] and sha256_file(stage) == record["sha256"], f"staged output identity mismatch: {name}")
        resolved[name] = (stage, final, record)
    published: list[str] = []
    outputs: dict[str, dict[str, Any]] = {}
    try:
        for name in order:
            stage, final, record = resolved[name]
            observed = publish_staged_create_if_absent(
                stage,
                final,
                expected_size=int(record["size_bytes"]),
                expected_sha256=str(record["sha256"]),
            )
            observed["path"] = FINAL_RELATIVES[name]
            outputs[name] = observed
            published.append(name)
    except Exception as exc:
        physically_published = [
            name for name in order if _lexists(resolved[name][1])
        ]
        if published or physically_published:
            raise PartialPublicationError(
                "partial V5 final publication is terminal; incident and V6 required",
                physically_published or published,
            ) from exc
        raise
    return outputs, tuple(published)


def validate_exact11_completion(
    artifact_root: Path,
    *,
    amendment: Mapping[str, Any] | None = None,
    expected_lock_payload: Mapping[str, Any] | None = None,
    expected_manifest_payload: Mapping[str, Any] | None = None,
    require_transaction_absent: bool = True,
) -> dict[str, dict[str, Any]]:
    root = path_under(artifact_root, OUTPUT_ROOT_RELATIVE.rstrip("/"), label="V5 output root")
    _require(root.is_dir() and not _linklike(root), "V5 output root absent/link-like")
    expected = set(EXACT11_BASENAMES)
    entries = tuple(root.iterdir())
    observed = {path.name for path in entries if path.is_file()}
    _require(observed == expected, f"V5 exact11 file set mismatch: {sorted(observed ^ expected)}")
    _require(
        all(not _linklike(root / name) for name in expected),
        "V5 exact11 contains a link-like path",
    )
    directories = {path.name for path in entries if path.is_dir()}
    transaction = root / ".transaction_v5"
    if require_transaction_absent:
        _require(not directories, "V5 completed namespace retains a directory")
        _require(not _lexists(transaction), "V5 completed transaction path still exists")
    else:
        _require(directories == {".transaction_v5"}, "V5 precleanup directory set mismatch")
        _require(not any(transaction.iterdir()), "V5 final transaction directory is not empty")
    contract = amendment or validate_amendment(artifact_root)
    validate_final_alias_schema_closure(artifact_root, contract)
    decision = strict_json_load(root / "TARGET_FREE_FAMILY_DECISION.json", label="V5 final decision")
    lock = strict_json_load(root / "TARGET_FREE_FAMILY_LOCK.json", label="V5 final family lock")
    _require(
        (root / "TARGET_FREE_FAMILY_LOCK.json").read_bytes()
        == pretty_json_bytes(lock),
        "V5 final family lock serialization mismatch",
    )
    validate_decision_payload(decision, contract)
    _require(set(lock) == LOCK_TOP_KEYS, "V5 final family-lock schema mismatch")
    decision_statuses = contract["output_contract"]["decision_status_literals_exact"]
    if decision["status"] == decision_statuses["positive"]:
        _require(
            decision["selected_family"]
            in contract["family_decision_and_salvage"]["fixed_priority_if_both_pass"]
            and decision["global_ambiguity"] is False,
            "V3 positive final decision mismatch",
        )
        _require(
            lock["status"] == LOCK_STATUS_BY_DECISION_STATUS[decision["status"]]
            and lock["selected_family"] == decision["selected_family"],
            "V3 positive family lock mismatch",
        )
    elif decision["status"] == decision_statuses["global_ambiguity"]:
        _require(
            decision["selected_family"] is None
            and decision["global_ambiguity"] is True
            and lock["status"] == LOCK_STATUS_BY_DECISION_STATUS[decision["status"]]
            and lock["selected_family"] is None,
            "V3 global-ambiguity tombstone mismatch",
        )
    else:
        _require(
            decision["status"] == decision_statuses["clear_no_family"]
            and decision["selected_family"] is None
            and decision["global_ambiguity"] is False
            and lock["status"] == LOCK_STATUS_BY_DECISION_STATUS[decision["status"]]
            and lock["selected_family"] is None,
            "V3 clear-no-family tombstone mismatch",
        )
    manifest = root / "TRACK_A_TARGET_FREE_RUN_MANIFEST.json"
    payload = strict_json_load(manifest, label="V5 run manifest")
    _require(
        manifest.read_bytes() == pretty_json_bytes(payload),
        "V5 final run-manifest serialization mismatch",
    )
    _require(set(payload) == MANIFEST_TOP_KEYS, "V5 run manifest schema mismatch")
    _require(payload.get("manifest_is_last_commit_marker") is True, "V5 manifest-last marker false")
    if expected_lock_payload is not None:
        _require_exact_tree(
            lock, expected_lock_payload,
            label="V3 final family-lock expected payload",
        )
    if expected_manifest_payload is not None:
        _require_exact_tree(
            payload, expected_manifest_payload,
            label="V3 final run-manifest expected payload",
        )
        _require(
            set(payload["alias_identities"]) == set(ALIAS_ORDER),
            "V3 final manifest alias identity set mismatch",
        )
        for name in ALIAS_ORDER:
            observed_alias = file_identity(root / name, relative_to=artifact_root)
            _require_exact_tree(
                payload["alias_identities"][name], observed_alias,
                label=f"V3 final manifest alias identity: {name}",
            )
        _require(
            set(payload["output_identities"]) == set(MANIFEST_OUTPUT_ORDER),
            "V3 final manifest output8 set mismatch",
        )
        for name in MANIFEST_OUTPUT_ORDER:
            recorded = payload["output_identities"][name]
            observed_output = file_identity(
                root / name, relative_to=artifact_root
            )
            _require_exact_tree(
                {key: recorded[key] for key in FILE_IDENTITY_KEYS},
                observed_output,
                label=f"V3 final manifest physical output identity: {name}",
            )
    return {name: file_identity(root / name, relative_to=artifact_root) for name in EXACT11_SCHEMA_ORDER}


def finalize_validated_authority_chain(
    artifact_root: Path,
    validation: Mapping[str, Any],
    *,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Build the two sealer JSONs, publish exact9 manifest-last, and recheck exact11."""

    amendment = validation.get("amendment")
    decision = validation.get("staged_decision")
    fit_ledger = validation.get("staged_fit_ledger")
    runner_stdout = validation.get("runner_stdout_payload")
    threadpool = validation.get("threadpool_info_before_inside_after")
    _require(isinstance(amendment, Mapping), "finalization amendment absent")
    _require(isinstance(decision, Mapping), "finalization staged decision absent")
    _require(isinstance(fit_ledger, Mapping), "finalization fit ledger absent")
    _require(isinstance(runner_stdout, Mapping), "finalization runner stdout absent")
    _require(isinstance(threadpool, Mapping), "finalization threadpool evidence absent")
    root = Path(os.path.abspath(artifact_root))
    parent_plan = load_bound_parent_plan(root, amendment)
    controls = validation["controls"]
    control_identities = validation["control_identities"]
    code_identities = validation["code_identities"]
    staged_output_identities = validation["staged_output_identities"]
    postrun_identity = control_identities["postrun_pass"]
    lock_payload = build_family_lock_payload(
        amendment=amendment,
        decision=decision,
        code_identities=code_identities,
        postrun_pass_identity=postrun_identity,
        parent_plan=parent_plan,
    )
    lock_identity = stage_sealer_json_payload(
        root, "TARGET_FREE_FAMILY_LOCK.json", lock_payload
    )
    manifest_created = created_utc or utc_now_exact()
    manifest_payload = build_run_manifest_payload(
        amendment=amendment,
        decision=decision,
        fit_ledger=fit_ledger,
        code_identities=code_identities,
        control_identities=control_identities,
        controls=controls,
        staged_output_identities=staged_output_identities,
        lock_output_identity=lock_identity,
        threadpool_info=threadpool,
        no_forbidden_access_attestation=runner_stdout[
            "no_forbidden_access_attestation"
        ],
        created_utc=manifest_created,
    )
    manifest_identity = stage_sealer_json_payload(
        root, "TRACK_A_TARGET_FREE_RUN_MANIFEST.json", manifest_payload
    )
    runner_staged = staged_output_identities["runner_staged_outputs"]
    staged_for_publication = {
        **{name: runner_staged[name] for name in RUNNER_PUBLICATION_ORDER},
        "TARGET_FREE_FAMILY_LOCK.json": lock_identity,
        "TRACK_A_TARGET_FREE_RUN_MANIFEST.json": manifest_identity,
    }
    published_outputs, published_order = publish_final_sequence(
        root, staged_for_publication
    )
    _require(
        published_order == FINAL_PUBLICATION_ORDER,
        "finalization publication order drift",
    )
    validate_exact11_completion(
        root,
        amendment=amendment,
        expected_lock_payload=lock_payload,
        expected_manifest_payload=manifest_payload,
        require_transaction_absent=False,
    )
    transaction = path_under(
        root,
        TRANSACTION_ROOT_RELATIVE.rstrip("/"),
        label="completed V5 transaction cleanup",
    )
    _require(transaction.is_dir() and not any(transaction.iterdir()), "completed V5 transaction is not empty before cleanup")
    transaction.rmdir()
    exact11 = validate_exact11_completion(
        root,
        amendment=amendment,
        expected_lock_payload=lock_payload,
        expected_manifest_payload=manifest_payload,
        require_transaction_absent=True,
    )
    return {
        "status": "PASS_V5_EXACT11_COMMITTED_MANIFEST_LAST",
        "terminal_state": decision["status"],
        "published_order": list(published_order),
        "published_outputs": published_outputs,
        "exact11_identities": exact11,
        "models_fit_by_sealer": 0,
        "parquet_logical_value_reads_by_sealer": 0,
        "network_requests_by_sealer": 0,
    }


def validate_csv_shape(
    path: Path,
    *,
    columns: Sequence[str],
    rows: int,
    label: str,
    invalid_reason_enum: Sequence[str] | None = None,
    boolean_columns: Mapping[str, bool] | None = None,
    boundary_flags_column: str | None = None,
    boundary_flag_order: Sequence[str] | None = None,
) -> None:
    _require(path.is_file() and not _linklike(path), f"{label} is absent/link-like")
    with path.open("rb") as binary:
        first = True
        for block in iter(lambda: binary.read(1 << 20), b""):
            if first:
                _require(not block.startswith(b"\xef\xbb\xbf"), f"{label} has a UTF-8 BOM")
                first = False
            _require(b"\r" not in block and b"\x00" not in block, f"{label} has forbidden CR/NUL bytes")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise TargetFreeSealError(f"{label} is empty") from exc
        _require(header == list(columns), f"{label} header mismatch")
        reason_index = header.index("invalid_reason") if "invalid_reason" in header else None
        boolean_columns = dict(boolean_columns or {})
        _require(
            all(
                isinstance(name, str)
                and name in header
                and type(nullable) is bool
                for name, nullable in boolean_columns.items()
            ),
            f"{label} boolean-column contract mismatch",
        )
        boolean_indices = {
            header.index(name): (name, nullable)
            for name, nullable in boolean_columns.items()
        }
        _require(
            (boundary_flags_column is None) == (boundary_flag_order is None),
            f"{label} boundary-field contract is incomplete",
        )
        boundary_index: int | None = None
        frozen_boundary_order: tuple[str, ...] = ()
        if boundary_flags_column is not None:
            _require(
                boundary_flags_column in header,
                f"{label} boundary-field column is absent",
            )
            assert boundary_flag_order is not None
            frozen_boundary_order = tuple(boundary_flag_order)
            _require(
                frozen_boundary_order
                and len(frozen_boundary_order) == len(set(frozen_boundary_order))
                and all(isinstance(name, str) and name for name in frozen_boundary_order),
                f"{label} frozen boundary order is invalid",
            )
            boundary_index = header.index(boundary_flags_column)
        allowed_reasons = (
            frozenset(invalid_reason_enum)
            if invalid_reason_enum is not None
            else None
        )
        if allowed_reasons is not None:
            _require(
                reason_index is not None,
                f"{label} invalid-reason enum supplied without a column",
            )
        count = 0
        for row in reader:
            _require(len(row) == len(header), f"{label} row width mismatch")
            for value in row:
                _require(value.strip().lower() not in NONFINITE_TEXT, f"{label} contains nonfinite text")
                try:
                    numeric = float(value)
                except ValueError:
                    continue
                _require(
                    math.isfinite(numeric),
                    f"{label} contains a numeric token that overflows to nonfinite",
                )
            if allowed_reasons is not None:
                assert reason_index is not None
                _require(
                    row[reason_index] == ""
                    or row[reason_index] in allowed_reasons,
                    f"{label} invalid_reason is outside the frozen exact enum",
                )
            for index, (name, nullable) in boolean_indices.items():
                value = row[index]
                _require(
                    value in {"TRUE", "FALSE"} or (nullable and value == ""),
                    f"{label} boolean cell is not uppercase TRUE/FALSE: {name}",
                )
            if boundary_index is not None:
                text = row[boundary_index]
                if text:
                    names = text.split(";")
                    _require(
                        all(names)
                        and len(names) == len(set(names))
                        and all(name in frozen_boundary_order for name in names)
                        and names
                        == [name for name in frozen_boundary_order if name in names],
                        f"{label} boundary flags are inactive, unknown, duplicate, or out of order",
                    )
            count += 1
    _require(count == rows, f"{label} row count mismatch")


def _require_ordered_unique_subset(
    value: Any,
    frozen_order: Sequence[str],
    *,
    label: str,
    allowed_empty: bool = True,
) -> list[str]:
    _require(
        isinstance(value, list) and all(isinstance(item, str) for item in value),
        f"{label} must be a string list",
    )
    if not allowed_empty:
        _require(bool(value), f"{label} must not be empty")
    index = {item: ordinal for ordinal, item in enumerate(frozen_order)}
    _require(all(item in index for item in value), f"{label} has an unfrozen member")
    _require(len(value) == len(set(value)), f"{label} has duplicate members")
    _require(
        [index[item] for item in value] == sorted(index[item] for item in value),
        f"{label} is not in frozen order",
    )
    return list(value)


def _require_bool_or_none(value: Any, *, label: str) -> bool | None:
    _require(value is None or isinstance(value, bool), f"{label} must be bool or null")
    return value


def _require_canonical_float_hex(value: Any, text: Any, *, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} decimal value is not numeric",
    )
    numeric = float(value)
    _require(math.isfinite(numeric), f"{label} decimal value is non-finite")
    _require(
        isinstance(text, str) and text == numeric.hex(),
        f"{label} float-hex witness mismatch",
    )
    return numeric


def _validate_tied_witnesses(
    witnesses: Any,
    *,
    responses: Sequence[str],
    raw_by_response: Mapping[str, Mapping[str, Any]],
) -> None:
    _require(isinstance(witnesses, list), "decision all_tied_witnesses is not a list")
    response_index = {name: ordinal for ordinal, name in enumerate(responses)}
    observed_order: list[tuple[int, int]] = []
    observed_keys: set[tuple[str, str]] = set()
    for ordinal, witness in enumerate(witnesses):
        _require(isinstance(witness, Mapping) and set(witness) == TIED_WITNESS_KEYS, f"tie witness schema mismatch: {ordinal}")
        kind = witness["tie_kind"]
        diagnostic = witness["diagnostic"]
        _require(kind in TIE_KINDS and diagnostic in response_index, f"tie witness kind/diagnostic mismatch: {ordinal}")
        key = (diagnostic, kind)
        _require(key not in observed_keys, f"duplicate tie witness: {key}")
        observed_keys.add(key)
        observed_order.append((response_index[diagnostic], TIE_KINDS.index(kind)))
        estimators = _require_ordered_unique_subset(
            witness["estimators"], ESTIMATOR_ENUM, label=f"tie estimators {ordinal}", allowed_empty=False
        )
        _require(tuple(estimators) == ESTIMATOR_ENUM, f"tie witness is not the exact two-estimator set: {ordinal}")
        metric = witness["metric_name"]
        _require(metric == ("RMSE" if kind == "INDEPENDENT_RMSE" else "R2"), f"tie metric mismatch: {ordinal}")
        hexes = witness["metric_values_float_hex"]
        _require(isinstance(hexes, list) and len(hexes) == len(estimators), f"tie metric witness length mismatch: {ordinal}")
        for position, text in enumerate(hexes):
            _require(isinstance(text, str), f"tie metric hex is not a string: {ordinal}/{position}")
            try:
                value = float.fromhex(text)
            except ValueError as exc:
                raise TargetFreeSealError(f"tie metric hex malformed: {ordinal}/{position}") from exc
            _require(math.isfinite(value) and value.hex() == text, f"tie metric hex noncanonical: {ordinal}/{position}")
        _require(
            len(hexes) == 2 and hexes[0] == hexes[1],
            f"tie metric witnesses are not bit-exactly equal: {ordinal}",
        )
        resolution = witness["resolution"]
        _require(resolution in TIE_RESOLUTIONS, f"tie resolution mismatch: {ordinal}")
        _require(isinstance(witness["global_ambiguity"], bool), f"tie global_ambiguity is not bool: {ordinal}")
        _require(
            witness["global_ambiguity"] is (resolution == "GLOBAL_AMBIGUITY"),
            f"tie global resolution mismatch: {ordinal}",
        )
        raw = raw_by_response[diagnostic]
        if kind == "INDEPENDENT_RMSE":
            _require(resolution == "GLOBAL_AMBIGUITY", f"independent RMSE tie did not stop globally: {ordinal}")
            _require(raw["classification"] == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE", f"independent tie/raw classification mismatch: {ordinal}")
        else:
            _require(raw["parent_reference_estimators"] == estimators, f"parent tie estimator witness mismatch: {ordinal}")
            if resolution == "GLOBAL_AMBIGUITY":
                _require(raw["classification"] == "AMBIGUOUS_PARENT_ESTIMATOR_TIE", f"parent global tie/raw classification mismatch: {ordinal}")
            elif resolution == "MASKED_BY_PRIOR_AFFINE_REDUNDANCY":
                _require(
                    raw["classification"] == "CLEAR_REDUNDANT_AFFINE"
                    and bool(raw["affine_qualifying_predictors"])
                    and raw["parent_redundancy_boolean"] is None,
                    f"masked parent tie/affine precedence mismatch: {ordinal}",
                )
            else:
                expected_boolean = resolution == "SHARED_PARENT_REDUNDANCY_TRUE"
                _require(raw["parent_redundancy_boolean"] is expected_boolean, f"parent shared tie boolean mismatch: {ordinal}")
    _require(observed_order == sorted(observed_order), "tie witnesses are not in response/independent-parent order")
    for response, raw in raw_by_response.items():
        if raw["classification"] == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE":
            _require((response, "INDEPENDENT_RMSE") in observed_keys, f"independent tie witness absent: {response}")
        parent_tie_present = (response, "PARENT_MAX_R2") in observed_keys
        _require(
            parent_tie_present
            is (len(raw["parent_reference_estimators"]) == 2),
            f"parent tie witness/reference cardinality mismatch: {response}",
        )


def _threshold_scope_order(
    cutoff: str,
    diagnostic: str,
    amendment: Mapping[str, Any],
) -> tuple[str, ...]:
    responses = amendment["planned_fit_budget"]["response_order"]
    predictors = amendment["input_and_join_contract"]["predictor_columns_exact"]
    endpoint = diagnostic in ENDPOINT_DIAGNOSTICS
    prefix = "ENDPOINT" if endpoint else "RAW"
    if cutoff.startswith("AFFINE_"):
        if endpoint:
            return ()
        return tuple(f"RAW/{diagnostic}/AFFINE/{predictor}" for predictor in predictors)
    if cutoff.startswith("PARENT_"):
        if endpoint:
            return ()
        return tuple(f"RAW/{diagnostic}/PARENT/{estimator}" for estimator in ESTIMATOR_ENUM)
    if cutoff.startswith("INDEPENDENT_"):
        if endpoint:
            if cutoff not in (
                "INDEPENDENT_R2_STABLE_MARGIN_0.9925",
                "INDEPENDENT_NRMSE_STABLE_MARGIN_0.025",
            ):
                return ()
            return (f"ENDPOINT/{diagnostic}/POOLED",)
        _require(diagnostic in responses, "raw independent threshold diagnostic mismatch")
        return tuple(f"RAW/{diagnostic}/INDEPENDENT/{estimator}" for estimator in ESTIMATOR_ENUM)
    if cutoff.startswith("CELL_"):
        labels = (
            ("YEAR", ("2022", "2023")),
            ("SEASON", ("DJF", "MAM", "JJA", "SON")),
            ("FORECAST_HOUR", ("EARLY", "MIDDLE", "LATE")),
            ("GROUP", ("kpx_group_1", "kpx_group_2", "kpx_group_3")),
            ("SITE", tuple(f"{site:02d}" for site in range(1, 18))),
        )
        return tuple(
            f"{prefix}/{diagnostic}/CELL/{kind}/{label}"
            for kind, values in labels
            for label in values
        )
    distribution_scopes = (
        "SITE_POOLED", "kpx_group_1", "kpx_group_2", "kpx_group_3"
    )
    if cutoff.startswith("MONTHLY_IQR_RATIO_"):
        return tuple(
            f"{prefix}/{diagnostic}/DISTRIBUTION/{scope}/MONTH/{month:02d}"
            for scope in distribution_scopes
            for month in range(1, 13)
        )
    if cutoff == "YEAR_PSI_0.5":
        return tuple(
            f"{prefix}/{diagnostic}/DISTRIBUTION/{scope}/YEAR/2022_2023"
            for scope in distribution_scopes
        )
    raise TargetFreeSealError(f"unknown threshold cutoff scope grammar: {cutoff}")


def validate_decision_payload(
    payload: Any, amendment: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the approved exact16 decision and its frozen nested closure."""

    _require(isinstance(payload, Mapping) and set(payload) == DECISION_TOP_KEYS, "decision top-level schema mismatch")
    _require(
        isinstance(payload.get("schema_version"), int)
        and not isinstance(payload["schema_version"], bool)
        and payload["schema_version"] == 5,
        "decision schema_version mismatch",
    )
    _require(
        payload.get("artifact_type") == "TARGET_FREE_FAMILY_DECISION_V5",
        "decision artifact_type mismatch",
    )
    responses = amendment["planned_fit_budget"]["response_order"]
    predictors = amendment["input_and_join_contract"]["predictor_columns_exact"]
    derived = amendment["derived_wind_contract"]["derived_order_exact"]
    families = amendment["family_decision_and_salvage"]["fixed_priority_if_both_pass"]
    cutoff_order = amendment["component_decision_truth_table"]["continuous_cutoffs_exact"]
    _require(tuple(cutoff_order) == tuple(CUTOFF_VALUES), "decision cutoff implementation drift")
    classifications = {
        row["classification"]: row["counts_as_nonredundant"]
        for row in amendment["component_decision_truth_table"]["rows"]
        if row["classification"] != "GLOBAL_INTEGRITY_FAILURE"
    }
    _require(payload["raw_component_order"] == responses, "decision raw component order mismatch")
    raw_results = payload["raw_component_results"]
    _require(isinstance(raw_results, list) and len(raw_results) == len(responses) == 9, "decision raw result count mismatch")
    raw_by_response: dict[str, Mapping[str, Any]] = {}
    for ordinal, (response, record) in enumerate(zip(responses, raw_results)):
        _require(isinstance(record, Mapping) and set(record) == RAW_DECISION_KEYS, f"raw decision schema mismatch: {ordinal}")
        _require(record["response"] == response, f"raw decision response order mismatch: {ordinal}")
        classification = record["classification"]
        _require(classification in classifications, f"raw decision classification mismatch: {response}")
        _require(isinstance(record["counts_as_nonredundant"], bool), f"raw nonredundant flag is not bool: {response}")
        _require(record["counts_as_nonredundant"] is classifications[classification], f"raw nonredundant/classification mismatch: {response}")
        selector_null = classification in (*PREMODEL_CLASSIFICATIONS, "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE")
        selector = record["independent_estimator"]
        _require(selector is None if selector_null else selector in ESTIMATOR_ENUM, f"raw independent estimator nullability/enum mismatch: {response}")
        parents = _require_ordered_unique_subset(
            record["parent_reference_estimators"], ESTIMATOR_ENUM, label=f"raw parent estimators {response}"
        )
        _require((len(parents) == 0) is selector_null, f"raw parent reference nullability mismatch: {response}")
        if not selector_null:
            _require(len(parents) in (1, 2), f"raw parent reference cardinality mismatch: {response}")
        affine = _require_ordered_unique_subset(
            record["affine_qualifying_predictors"], predictors, label=f"raw affine hits {response}"
        )
        if selector_null:
            _require(not affine, f"raw affine hits must be empty before selector: {response}")
        if classification == "CLEAR_REDUNDANT_AFFINE":
            _require(bool(affine), f"raw affine redundancy classification lacks a qualifying hit: {response}")
        ambiguous = classification in AMBIGUOUS_CLASSIFICATIONS
        gates_null = classification in PREMODEL_CLASSIFICATIONS or ambiguous
        parent_boolean = _require_bool_or_none(
            record["parent_redundancy_boolean"],
            label=f"raw parent_redundancy_boolean {response}",
        )
        masked_affine_parent_tie = (
            classification == "CLEAR_REDUNDANT_AFFINE"
            and parent_boolean is None
        )
        _require(
            not masked_affine_parent_tie or len(parents) == 2,
            f"masked affine parent tie lacks exact two-estimator references: {response}",
        )
        _require(
            (parent_boolean is None) is (gates_null or masked_affine_parent_tie),
            f"raw parent_redundancy_boolean nullability mismatch: {response}",
        )
        for field in ("cell_stability_pass", "distribution_stability_pass"):
            value = _require_bool_or_none(record[field], label=f"raw {field} {response}")
            _require((value is None) is gates_null, f"raw {field} nullability mismatch: {response}")
        flags = _require_ordered_unique_subset(
            record["boundary_equality_flags"], cutoff_order, label=f"raw boundary flags {response}"
        )
        _require(bool(flags) is (classification == "AMBIGUOUS_THRESHOLD_EQUALITY"), f"raw threshold flags/classification mismatch: {response}")
        reason = record["ambiguity_reason"]
        _require(reason == classification if ambiguous else reason is None, f"raw ambiguity reason mismatch: {response}")
        raw_by_response[response] = record

    endpoints = payload["endpoint_veto_results"]
    _require(isinstance(endpoints, list) and len(endpoints) == len(ENDPOINT_DIAGNOSTICS), "decision endpoint result count mismatch")
    endpoint_by_diagnostic: dict[str, Mapping[str, Any]] = {}
    for ordinal, (diagnostic, record) in enumerate(zip(ENDPOINT_DIAGNOSTICS, endpoints)):
        _require(isinstance(record, Mapping) and set(record) == ENDPOINT_DECISION_KEYS, f"endpoint decision schema mismatch: {ordinal}")
        _require(record["diagnostic"] == diagnostic, f"endpoint diagnostic order mismatch: {ordinal}")
        verdict = record["verdict"]
        _require(verdict in ENDPOINT_VERDICTS, f"endpoint verdict mismatch: {diagnostic}")
        binding = record["source_component_estimator_binding"]
        expected_sources = ENDPOINT_SOURCE_RESPONSES[diagnostic]
        _require(isinstance(binding, Mapping) and set(binding) == set(expected_sources), f"endpoint source binding schema mismatch: {diagnostic}")
        for source in expected_sources:
            _require(binding[source] == raw_by_response[source]["independent_estimator"], f"endpoint source estimator mismatch: {diagnostic}/{source}")
        source_premodel = any(raw_by_response[source]["classification"] in PREMODEL_CLASSIFICATIONS for source in expected_sources)
        source_tie = any(raw_by_response[source]["classification"] == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE" for source in expected_sources)
        gates = [
            _require_bool_or_none(record[field], label=f"endpoint {field} {diagnostic}")
            for field in (
                "pooled_stable_margin_pass", "cell_stability_pass",
                "distribution_stability_pass",
            )
        ]
        flags = _require_ordered_unique_subset(
            record["boundary_equality_flags"], cutoff_order, label=f"endpoint boundary flags {diagnostic}"
        )
        ambiguous = verdict in ("AMBIGUOUS_THRESHOLD_EQUALITY", "AMBIGUOUS_RAW_ESTIMATOR_TIE")
        _require(record["ambiguity_reason"] == verdict if ambiguous else record["ambiguity_reason"] is None, f"endpoint ambiguity reason mismatch: {diagnostic}")
        _require(isinstance(record["clear_pass"], bool) and record["clear_pass"] is (verdict == "PASS_ENDPOINT_NONREDUNDANT_MARGIN"), f"endpoint clear_pass/verdict mismatch: {diagnostic}")
        if source_tie:
            _require(verdict == "AMBIGUOUS_RAW_ESTIMATOR_TIE" and all(value is None for value in gates), f"endpoint raw-tie closure mismatch: {diagnostic}")
        elif source_premodel:
            _require(verdict == "CLEAR_ENDPOINT_VETO_FAILURE" and all(value is None for value in gates), f"endpoint premodel closure mismatch: {diagnostic}")
        elif verdict == "PASS_ENDPOINT_NONREDUNDANT_MARGIN":
            _require(all(value is True for value in gates), f"endpoint pass gates mismatch: {diagnostic}")
        elif verdict == "CLEAR_ENDPOINT_VETO_FAILURE":
            _require(all(isinstance(value, bool) for value in gates) and any(value is False for value in gates), f"endpoint clear-failure gates mismatch: {diagnostic}")
        else:
            _require(verdict == "AMBIGUOUS_THRESHOLD_EQUALITY", f"endpoint ambiguity closure mismatch: {diagnostic}")
            _require(any(value is None for value in gates), f"endpoint threshold equality did not null the affected aggregate: {diagnostic}")
        _require(bool(flags) is (verdict == "AMBIGUOUS_THRESHOLD_EQUALITY"), f"endpoint threshold flags/verdict mismatch: {diagnostic}")
        endpoint_by_diagnostic[diagnostic] = record

    witnesses = payload["all_threshold_float_hex_witnesses"]
    _require(isinstance(witnesses, list), "decision threshold witnesses are not a list")
    diagnostic_order = tuple(responses) + ENDPOINT_DIAGNOSTICS
    diagnostic_index = {name: ordinal for ordinal, name in enumerate(diagnostic_order)}
    cutoff_index = {name: ordinal for ordinal, name in enumerate(cutoff_order)}
    witness_keys: set[tuple[str, str, str]] = set()
    boundary_by_diagnostic: dict[str, set[str]] = {name: set() for name in diagnostic_order}
    previous_sort_key: tuple[int, int, int] | None = None
    for ordinal, witness in enumerate(witnesses):
        _require(isinstance(witness, Mapping) and set(witness) == THRESHOLD_WITNESS_KEYS, f"threshold witness schema mismatch: {ordinal}")
        cutoff = witness["cutoff_name"]
        diagnostic = witness["diagnostic"]
        scope = witness["scope"]
        _require(cutoff in cutoff_index and diagnostic in diagnostic_index, f"threshold witness cutoff/diagnostic mismatch: {ordinal}")
        scope_order = _threshold_scope_order(cutoff, diagnostic, amendment)
        _require(isinstance(scope, str) and scope in scope_order, f"threshold witness scope invalid: {ordinal}")
        key = (cutoff, scope, diagnostic)
        _require(key not in witness_keys, f"duplicate threshold witness: {key}")
        witness_keys.add(key)
        value = _require_canonical_float_hex(witness["value"], witness["value_float_hex"], label=f"threshold witness {ordinal}")
        cutoff_value = CUTOFF_VALUES[cutoff]
        _require(witness["cutoff_float_hex"] == cutoff_value.hex(), f"threshold cutoff hex mismatch: {ordinal}")
        _require(isinstance(witness["boundary_equal"], bool), f"threshold boundary flag is not bool: {ordinal}")
        _require(witness["boundary_equal"] is (value == cutoff_value), f"threshold boundary equality mismatch: {ordinal}")
        sort_key = (
            cutoff_index[cutoff],
            diagnostic_index[diagnostic],
            scope_order.index(scope),
        )
        _require(previous_sort_key is None or previous_sort_key < sort_key, "threshold witnesses are not in explicit frozen order")
        previous_sort_key = sort_key
        if witness["boundary_equal"]:
            boundary_by_diagnostic[diagnostic].add(cutoff)
    for diagnostic, record in {**raw_by_response, **endpoint_by_diagnostic}.items():
        expected_flags = [name for name in cutoff_order if name in boundary_by_diagnostic[diagnostic]]
        _require(record["boundary_equality_flags"] == expected_flags, f"decision boundary witness/flag mismatch: {diagnostic}")

    _validate_tied_witnesses(
        payload["all_tied_witnesses"], responses=responses, raw_by_response=raw_by_response
    )

    family_results = payload["family_clear_pass_results"]
    _require(isinstance(family_results, Mapping) and tuple(family_results) == tuple(families), "decision family clear-pass schema/order mismatch")
    _require(all(isinstance(value, bool) for value in family_results.values()), "decision family clear-pass value is not bool")
    pbl_pass = raw_by_response["HPBL_surface"]["counts_as_nonredundant"] is True
    wind_records = [raw_by_response[name] for name in responses if name != "HPBL_surface"]
    wind_physical = all(
        record["classification"] != "CLEAR_PHYSICAL_OR_COVERAGE_FAILURE"
        for record in wind_records
    )
    wind_unambiguous = all(
        record["classification"] not in AMBIGUOUS_CLASSIFICATIONS
        for record in wind_records
    )
    wind_nonredundant = [record["response"] for record in wind_records if record["counts_as_nonredundant"]]
    wind_levels = {
        int(name.split("_", 1)[1][:-2]) for name in wind_nonredundant
    }
    wind_pass = (
        wind_physical
        and wind_unambiguous
        and len(wind_nonredundant) >= 4
        and any(name.startswith("UGRD_") for name in wind_nonredundant)
        and any(name.startswith("VGRD_") for name in wind_nonredundant)
        and len(wind_levels) >= 2
        and all(record["clear_pass"] for record in endpoints)
    )
    _require(
        dict(family_results)
        == {"LOW_LEVEL_ISOBARIC_WIND_PROFILE": wind_pass, "PBL_HEIGHT": pbl_pass},
        "decision family clear-pass recomputation mismatch",
    )
    _require(payload["fixed_priority"] == families, "decision fixed priority mismatch")

    expected_reasons: list[str] = []
    expected_reasons.extend(
        f"AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE:{record['response']}"
        for record in raw_results
        if record["classification"] == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
    )
    expected_reasons.extend(
        "AMBIGUOUS_THRESHOLD_EQUALITY:"
        f"{witness['cutoff_name']}:{witness['scope']}:"
        f"{witness['diagnostic']}:{witness['value_float_hex']}"
        for witness in witnesses
        if witness["boundary_equal"]
    )
    expected_reasons.extend(
        f"AMBIGUOUS_PARENT_ESTIMATOR_TIE:{record['response']}"
        for record in raw_results
        if record["classification"] == "AMBIGUOUS_PARENT_ESTIMATOR_TIE"
    )
    expected_reasons.extend(
        f"AMBIGUOUS_GRAY_ZONE:{record['response']}"
        for record in raw_results
        if record["classification"] == "AMBIGUOUS_GRAY_ZONE"
    )
    _require(payload["global_ambiguity_reasons"] == expected_reasons, "decision global ambiguity reason order/content mismatch")
    global_ambiguity = bool(expected_reasons)
    _require(payload["global_ambiguity"] is global_ambiguity, "decision global_ambiguity mismatch")
    statuses = amendment["output_contract"]["decision_status_literals_exact"]
    expected_status = (
        statuses["global_ambiguity"]
        if global_ambiguity
        else statuses["positive"]
        if any(family_results.values())
        else statuses["clear_no_family"]
    )
    _require(payload["status"] == expected_status, "decision terminal status mismatch")
    selected = payload["selected_family"]
    selected_raw = payload["selected_raw_columns"]
    selected_derived = payload["selected_derived_columns"]
    if expected_status == statuses["positive"]:
        expected_family = next(family for family in families if family_results[family])
        _require(selected == expected_family, "decision selected family violates frozen priority")
        expected_raw = ["HPBL_surface"] if selected == "PBL_HEIGHT" else [name for name in responses if name != "HPBL_surface"]
        expected_derived = [] if selected == "PBL_HEIGHT" else derived
        _require(selected_raw == expected_raw and selected_derived == expected_derived, "decision selected column closure mismatch")
    else:
        _require(selected is None and selected_raw == [] and selected_derived == [], "decision negative/ambiguity selected fields mismatch")
    _require_exact_tree(
        payload["no_label_network_future_year_model_submission_attestation"],
        DECISION_FORBIDDEN_ACCESS_ATTESTATION,
        label="decision forbidden-access attestation",
    )
    return dict(payload)


def validate_fit_ledger_payload(
    ledger: Any,
    amendment: Mapping[str, Any],
    *,
    expected_status: str | None = None,
) -> dict[str, Any]:
    """Bind all 72 primary and 1,260 analytic slots without fitting anything."""

    _require(isinstance(ledger, Mapping) and set(ledger) == FIT_LEDGER_TOP_KEYS, "fit ledger top schema mismatch")
    _require(
        isinstance(ledger.get("schema_version"), int)
        and not isinstance(ledger["schema_version"], bool)
        and ledger["schema_version"] == 5,
        "fit ledger schema_version mismatch",
    )
    _require(
        ledger.get("artifact_type") == "TARGET_FREE_FIT_LEDGER_V5",
        "fit ledger artifact_type mismatch",
    )
    decision_statuses = amendment["output_contract"][
        "decision_status_literals_exact"
    ]
    allowed_statuses = tuple(decision_statuses.values())
    _require(
        isinstance(ledger.get("status"), str)
        and ledger["status"] in allowed_statuses,
        "fit ledger status is not a frozen decision status",
    )
    if expected_status is not None:
        _require(
            ledger["status"] == expected_status,
            "fit ledger status/decision crosslink mismatch",
        )
    slots = ledger.get("unit_slots")
    _require(isinstance(slots, list) and len(slots) == 1_332, "fit ledger slot count mismatch")
    budget = amendment["planned_fit_budget"]
    responses = budget["response_order"]
    predictors = amendment["input_and_join_contract"]["predictor_columns_exact"]
    invalid_reasons = frozenset(
        amendment["metric_primitives"]["invalid_or_skip_reason_enum_exact"]
    )
    primary_keys = set(budget["fit_ledger_record_schemas"]["primary_model_unit_record_exact_keys"])
    affine_keys = set(budget["fit_ledger_record_schemas"]["analytic_affine_unit_record_exact_keys"])
    completed_primary = 0
    completed_affine = 0
    skipped_primary = 0
    skipped_affine = 0
    observed_internal = {
        "sklearn.pipeline.Pipeline.fit": 0,
        "sklearn.preprocessing.StandardScaler.fit": 0,
        "sklearn.linear_model.Ridge.fit": 0,
        "sklearn.ensemble.ExtraTreesRegressor.fit": 0,
    }
    identifiers: set[str] = set()
    for index, slot in enumerate(slots):
        _require(isinstance(slot, Mapping), f"fit ledger slot is not an object: {index}")
        primary = index < 72
        _require(set(slot) == (primary_keys if primary else affine_keys), f"fit ledger slot schema mismatch: {index}")
        _require(
            isinstance(slot.get("decision_fit_ordinal"), int)
            and not isinstance(slot["decision_fit_ordinal"], bool)
            and slot["decision_fit_ordinal"] == index + 1,
            f"fit ledger decision ordinal mismatch: {index}",
        )
        identifier = slot.get("planned_unit_slot_id")
        _require(isinstance(identifier, str) and identifier not in identifiers, f"fit ledger slot id duplicate/invalid: {index}")
        identifiers.add(identifier)
        status = slot.get("unit_status")
        _require(status in ("COMPLETED", "SKIPPED_PREMODEL_CLEAR_VETO"), f"fit ledger status mismatch: {index}")
        completed = status == "COMPLETED"
        _require(slot.get("fit_completed") is completed, f"fit ledger completed flag mismatch: {index}")
        _require(
            slot.get("skip_reason") is None
            if completed
            else slot.get("skip_reason") in invalid_reasons,
            f"fit ledger skip reason is outside the frozen exact enum: {index}",
        )
        if primary:
            response_index, remainder = divmod(index, 8)
            fold_index, estimator_index = divmod(remainder, 2)
            expected_id = f"PMU__{response_index + 1:02d}__{fold_index + 1}__{estimator_index + 1}"
            estimator = ESTIMATOR_ENUM[estimator_index]
            _require(
                slot.get("unit_kind") == "PRIMARY_MODEL"
                and slot.get("response") == responses[response_index]
                and slot.get("fold") == FOLD_ENUM[fold_index]
                and slot.get("estimator_unit") == estimator
                and identifier == expected_id,
                f"fit ledger primary frozen order mismatch: {index}",
            )
            _require(
                slot.get("training_rows") == 14_688
                and type(slot["training_rows"]) is int
                and slot.get("training_columns") == 35
                and type(slot["training_columns"]) is int
                and slot.get("heldout_rows") == 4_896
                and type(slot["heldout_rows"]) is int,
                f"fit ledger primary fixed shape mismatch: {index}",
            )
            _require(
                slot.get("input_dtype")
                == (
                    "float64_C_CONTIGUOUS"
                    if estimator == "RIDGE_PIPELINE"
                    else "float32_C_CONTIGUOUS"
                )
                and slot.get("response_dtype") == "float64"
                and slot.get("random_state")
                == (None if estimator == "RIDGE_PIPELINE" else 260_810),
                f"fit ledger primary dtype/random-state mismatch: {index}",
            )
            call_fields = {
                "pipeline_fit_calls": 1 if completed and estimator == "RIDGE_PIPELINE" else 0,
                "standard_scaler_fit_calls": 1 if completed and estimator == "RIDGE_PIPELINE" else 0,
                "ridge_fit_calls": 1 if completed and estimator == "RIDGE_PIPELINE" else 0,
                "extra_trees_fit_calls": 1 if completed and estimator == "EXTRA_TREES" else 0,
                "predict_calls": 1 if completed else 0,
            }
            for field, expected in call_fields.items():
                _require(
                    isinstance(slot.get(field), int)
                    and not isinstance(slot[field], bool)
                    and slot[field] == expected,
                    f"fit ledger primary call count mismatch: {index}/{field}",
                )
            if completed:
                completed_primary += 1
                if estimator == "RIDGE_PIPELINE":
                    observed_internal["sklearn.pipeline.Pipeline.fit"] += 1
                    observed_internal["sklearn.preprocessing.StandardScaler.fit"] += 1
                    observed_internal["sklearn.linear_model.Ridge.fit"] += 1
                else:
                    observed_internal["sklearn.ensemble.ExtraTreesRegressor.fit"] += 1
            else:
                skipped_primary += 1
        else:
            affine_index = index - 72
            response_index, remainder = divmod(affine_index, 35 * 4)
            predictor_index, fold_index = divmod(remainder, 4)
            expected_id = f"AAU__{response_index + 1:02d}__{predictor_index + 1:02d}__{fold_index + 1}"
            _require(
                slot.get("unit_kind") == "ANALYTIC_AFFINE"
                and slot.get("response") == responses[response_index]
                and slot.get("predictor") == predictors[predictor_index]
                and slot.get("fold") == FOLD_ENUM[fold_index]
                and isinstance(slot.get("analytic_affine_ordinal"), int)
                and not isinstance(slot["analytic_affine_ordinal"], bool)
                and slot["analytic_affine_ordinal"] == affine_index + 1
                and identifier == expected_id,
                f"fit ledger affine frozen order mismatch: {index}",
            )
            _require(
                slot.get("training_rows") == 14_688
                and type(slot["training_rows"]) is int
                and slot.get("heldout_rows") == 4_896
                and type(slot["heldout_rows"]) is int,
                f"fit ledger affine fixed shape mismatch: {index}",
            )
            for base in ("sxx", "slope", "intercept"):
                value = slot.get(base)
                witness = slot.get(f"{base}_float_hex")
                if value is None:
                    _require(witness is None, f"fit ledger affine null/hex mismatch: {index}/{base}")
                else:
                    _require_canonical_float_hex(value, witness, label=f"fit ledger affine {index}/{base}")
            if completed:
                _require(
                    all(slot.get(base) is not None for base in ("sxx", "slope", "intercept"))
                    and isinstance(slot.get("constant_predictor_branch"), bool),
                    f"fit ledger completed affine numeric closure mismatch: {index}",
                )
            else:
                _require(
                    all(
                        slot.get(field) is None
                        for field in (
                            "sxx", "sxx_float_hex", "slope", "slope_float_hex",
                            "intercept", "intercept_float_hex",
                            "constant_predictor_branch",
                        )
                    ),
                    f"fit ledger skipped affine numeric closure mismatch: {index}",
                )
            if completed:
                completed_affine += 1
            else:
                skipped_affine += 1
    _require(completed_primary + skipped_primary == 72, "fit ledger primary accounting mismatch")
    _require(completed_affine + skipped_affine == 1_260, "fit ledger affine accounting mismatch")
    _require_exact_tree(
        ledger.get("planned_counts"), FIT_LEDGER_PLANNED_COUNTS,
        label="fit ledger planned_counts",
    )
    executed = ledger.get("executed_counts")
    skipped = ledger.get("skipped_counts")
    internal = ledger.get("internal_method_call_counts")
    _require(
        isinstance(executed, Mapping)
        and set(executed) == FIT_LEDGER_EXECUTED_KEYS,
        "fit ledger executed_counts schema mismatch",
    )
    _require(
        isinstance(skipped, Mapping)
        and set(skipped) == FIT_LEDGER_SKIPPED_KEYS,
        "fit ledger skipped_counts schema mismatch",
    )
    _require(
        isinstance(internal, Mapping)
        and set(internal) == FIT_LEDGER_INTERNAL_METHOD_KEYS,
        "fit ledger internal_method_call_counts schema mismatch",
    )
    for label, counts in (
        ("executed_counts", executed),
        ("skipped_counts", skipped),
        ("internal_method_call_counts", internal),
    ):
        _require(
            all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in counts.values()
            ),
            f"fit ledger {label} has a non-integer count",
        )
    expected_executed = {
        "completed_primary_model_units": completed_primary,
        "completed_analytic_affine_units": completed_affine,
        "completed_total_decision_units": completed_primary + completed_affine,
    }
    expected_skipped = {
        "skipped_primary_model_units": skipped_primary,
        "skipped_analytic_affine_units": skipped_affine,
        "skipped_total_decision_units": skipped_primary + skipped_affine,
    }
    _require_exact_tree(
        executed, expected_executed,
        label="fit ledger executed_counts/slot accounting",
    )
    _require_exact_tree(
        skipped, expected_skipped,
        label="fit ledger skipped_counts/slot accounting",
    )
    _require(
        executed["completed_primary_model_units"]
        + skipped["skipped_primary_model_units"]
        == FIT_LEDGER_PLANNED_COUNTS["primary_model_units"],
        "fit ledger primary completed+skipped equation mismatch",
    )
    _require(
        executed["completed_analytic_affine_units"]
        + skipped["skipped_analytic_affine_units"]
        == FIT_LEDGER_PLANNED_COUNTS["analytic_affine_units"],
        "fit ledger affine completed+skipped equation mismatch",
    )
    _require(
        executed["completed_total_decision_units"]
        + skipped["skipped_total_decision_units"]
        == FIT_LEDGER_PLANNED_COUNTS["total_decision_units"],
        "fit ledger total completed+skipped equation mismatch",
    )
    for key, expected in observed_internal.items():
        _require(internal[key] == expected, f"fit ledger internal method count mismatch: {key}")
    _require(
        internal["total_method_calls"]
        == sum(internal[key] for key in observed_internal),
        "fit ledger total internal method call count mismatch",
    )
    amendment_zero_counts = amendment["output_contract"][
        "output_record_schemas_exact"
    ]["TARGET_FREE_FIT_LEDGER_V3_json"]["zero_fit_counts"]
    _require_exact_tree(
        ledger.get("zero_fit_counts"), FIT_LEDGER_ZERO_COUNTS,
        label="fit ledger zero-fit counts",
    )
    _require_exact_tree(
        ledger.get("zero_fit_counts"), amendment_zero_counts,
        label="fit ledger amendment zero-fit counts",
    )
    return dict(ledger)


def validate_distribution_payload(
    distribution: Any,
    amendment: Mapping[str, Any],
    *,
    expected_status: str,
) -> dict[str, Any]:
    """Bind the frozen distribution materializer envelope and exact records."""

    _require(
        isinstance(distribution, Mapping)
        and set(distribution) == DISTRIBUTION_TOP_KEYS,
        "distribution top schema mismatch",
    )
    _require(
        isinstance(distribution.get("schema_version"), int)
        and not isinstance(distribution["schema_version"], bool)
        and distribution["schema_version"] == 5,
        "distribution schema_version mismatch",
    )
    _require(
        distribution.get("artifact_type")
        == "TARGET_FREE_DISTRIBUTION_STABILITY_V5",
        "distribution artifact_type mismatch",
    )
    _require(
        distribution.get("status") == expected_status
        and expected_status
        in amendment["output_contract"]["decision_status_literals_exact"].values(),
        "distribution status/decision crosslink mismatch",
    )
    _require(
        distribution.get("scope_order")
        == amendment["distribution_stability_contract"]["decision_scopes_exact"],
        "distribution scope order mismatch",
    )
    expected_diagnostics = [
        *amendment["planned_fit_budget"]["response_order"],
        *ENDPOINT_DIAGNOSTICS,
    ]
    _require(
        distribution.get("diagnostic_order") == expected_diagnostics,
        "distribution diagnostic order mismatch",
    )
    _require(
        isinstance(distribution.get("records"), list)
        and len(distribution["records"]) == 624,
        "distribution record count mismatch",
    )
    _require(
        isinstance(distribution.get("endpoint_pooled_metric_records"), list)
        and len(distribution["endpoint_pooled_metric_records"]) == 3,
        "endpoint pooled record count mismatch",
    )
    schemas = amendment["output_contract"]["output_record_schemas_exact"]
    record_keys = set(
        schemas["TARGET_FREE_DISTRIBUTION_STABILITY_V3_json"][
            "record_keys_exact"
        ]
    )
    endpoint_keys = set(
        schemas["TARGET_FREE_DISTRIBUTION_STABILITY_V3_json"][
            "endpoint_pooled_metric_record_keys_exact"
        ]
    )
    invalid_reasons = frozenset(
        amendment["metric_primitives"]["invalid_or_skip_reason_enum_exact"]
    )
    _require(
        all(
            isinstance(row, Mapping) and set(row) == record_keys
            for row in distribution["records"]
        ),
        "distribution record schema mismatch",
    )
    _require(
        all(
            isinstance(row, Mapping) and set(row) == endpoint_keys
            for row in distribution["endpoint_pooled_metric_records"]
        ),
        "endpoint pooled schema mismatch",
    )
    for row in (
        *distribution["records"],
        *distribution["endpoint_pooled_metric_records"],
    ):
        _require(
            row["invalid_reason"] is None
            or row["invalid_reason"] in invalid_reasons,
            "distribution invalid_reason is outside the frozen exact enum",
        )
    return dict(distribution)


def validate_staged_json_outputs(
    artifact_root: Path, amendment: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    decision = strict_json_load(
        path_under(artifact_root, RUNNER_STAGED_RELATIVES["TARGET_FREE_FAMILY_DECISION.json"], label="staged decision"),
        label="staged target-free decision",
    )
    validate_decision_payload(decision, amendment)
    ledger = strict_json_load(
        path_under(artifact_root, RUNNER_STAGED_RELATIVES["TARGET_FREE_FIT_LEDGER_V5.json"], label="staged fit ledger"),
        label="staged fit ledger",
    )
    validate_fit_ledger_payload(
        ledger, amendment, expected_status=decision["status"]
    )
    distribution = strict_json_load(
        path_under(artifact_root, RUNNER_STAGED_RELATIVES["TARGET_FREE_DISTRIBUTION_STABILITY_V5.json"], label="staged distribution"),
        label="staged distribution stability",
    )
    validate_distribution_payload(
        distribution, amendment, expected_status=decision["status"]
    )
    return {
        "decision": decision,
        "fit_ledger": ledger,
        "distribution": distribution,
    }


def validate_staged_csv_outputs(artifact_root: Path, amendment: Mapping[str, Any]) -> None:
    schemas = amendment["output_contract"]["output_record_schemas_exact"]
    invalid_reasons = amendment["metric_primitives"][
        "invalid_or_skip_reason_enum_exact"
    ]
    cutoff_order = tuple(
        amendment["component_decision_truth_table"]["continuous_cutoffs_exact"]
    )
    _require(
        cutoff_order == tuple(CUTOFF_VALUES),
        "V4 CSV boundary cutoff order drift",
    )
    contracts = (
        (
            "TARGET_FREE_DUPLICATE_METRICS.csv",
            "TARGET_FREE_DUPLICATE_METRICS_csv",
            {
                "valid": False,
                "independent_selected": False,
                "parent_max_r2_witness": False,
                "parent_redundancy_boolean": True,
                "affine_qualifies": False,
            },
            "boundary_equality_flags",
        ),
        (
            "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv",
            "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE_csv",
            {
                "valid": False,
                "r2_boundary_equal_0_995": False,
                "nrmse_boundary_equal_0_02": False,
                "weak_floor_applicable": False,
                "nrmse_boundary_equal_0_01": False,
                "strict_not_exact_pass": False,
                "weak_floor_pass": True,
            },
            None,
        ),
    )
    for name, schema_key, boolean_columns, boundary_column in contracts:
        validate_csv_shape(
            path_under(artifact_root, RUNNER_STAGED_RELATIVES[name], label=f"staged CSV {name}"),
            columns=schemas[schema_key]["columns_exact"],
            rows=int(schemas[schema_key]["row_slots_exact"]),
            label=name,
            invalid_reason_enum=invalid_reasons,
            boolean_columns=boolean_columns,
            boundary_flags_column=boundary_column,
            boundary_flag_order=cutoff_order if boundary_column is not None else None,
        )


def _network_forbidden(*_args: Any, **_kwargs: Any) -> Any:
    raise TargetFreeSealError("network access is forbidden in the V5 sealer")


_NETWORK_GUARD_INSTALLED = False


def _network_audit_guard(event: str, _args: tuple[Any, ...]) -> None:
    if event in {
        "socket.__new__", "socket.bind", "socket.connect", "socket.connect_ex",
        "socket.getaddrinfo", "socket.gethostbyaddr", "socket.gethostbyname",
        "socket.getnameinfo", "socket.sendmsg", "socket.sendto",
    }:
        _network_forbidden()


def install_network_guard() -> None:
    global _NETWORK_GUARD_INSTALLED
    if not _NETWORK_GUARD_INSTALLED:
        sys.addaudithook(_network_audit_guard)
        _NETWORK_GUARD_INSTALLED = True
    for name in (
        "create_connection", "getaddrinfo", "gethostbyaddr", "gethostbyname", "getnameinfo"
    ):
        setattr(socket, name, _network_forbidden)


def static_self_validate(source_path: Path | None = None) -> dict[str, Any]:
    path = Path(source_path or __file__).resolve()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden_imports = {
        "numpy", "pandas", "scipy", "sklearn", "joblib",
        "threadpoolctl", "eccodes", "requests", "urllib", "http", "subprocess",
    }
    _require(imports.isdisjoint(forbidden_imports), f"sealer imports forbidden package: {sorted(imports & forbidden_imports)}")
    arrow_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.startswith("pyarrow")
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("pyarrow")
    }
    _require(
        arrow_imports == {"pyarrow", "pyarrow.parquet"},
        "sealer PyArrow imports exceed the exact footer-only pair",
    )
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    _require(calls.count("os.link") == 1, "sealer must contain exactly one low-level hard-link publisher")
    _require(not any(call in {"os.replace", "os.rename", "Path.rename", "Path.replace"} for call in calls), "overwrite-capable publication call present")
    _require(not any(call.endswith(".fit") or call.endswith(".predict") for call in calls), "sealer contains model fit/predict call")
    _require(
        calls.count("pq.read_metadata") == 1,
        "sealer must contain exactly one footer-only Parquet metadata call",
    )
    _require(
        not any(
            call.endswith("read_parquet")
            or call.endswith("read_table")
            or call.endswith("ParquetFile")
            or call.endswith("read_row_group")
            for call in calls
        ),
        "sealer contains Parquet logical-value route",
    )
    _require(not any(call.startswith("socket.") and call not in {"socket." + name for name in ()} for call in calls), "sealer contains direct socket call")
    required_functions = {
        "validate_amendment", "validate_amendment_v4",
        "validate_sealer_runtime_identity",
        "validate_control_payload", "validate_auditor_stdout",
        "validate_actual_execution_argv", "validate_authority_chain",
        "validate_postrun_namespace_state",
        "validate_decision_payload", "validate_fit_ledger_payload",
        "validate_distribution_payload",
        "build_family_lock_payload", "validate_family_lock_payload",
        "build_run_manifest_payload", "validate_run_manifest_payload",
        "validate_final_alias_schema_closure",
        "publish_staged_create_if_absent", "publish_alias_pair_predata",
        "publish_final_sequence", "validate_exact11_completion",
        "finalize_validated_authority_chain", "static_self_validate",
    }
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    _require(required_functions.issubset(functions), "sealer required function closure incomplete")
    return {
        "status": "PASS_V5_SEALER_STATIC_SELF_VALIDATION",
        "source": file_identity(path),
        "single_low_level_publisher": True,
        "os_link_call_count": 1,
        "overwrite_capable_calls": 0,
        "network_calls": 0,
        "parquet_logical_value_reads": 0,
        "parquet_footer_metadata_reads": 1,
        "models_fit": 0,
        "predictions_computed": 0,
    }


def publish_code_seal(
    artifact_root: Path,
    payload: Mapping[str, Any],
    *,
    amendment: Mapping[str, Any],
    repo_root: Path = REPO,
) -> dict[str, Any]:
    validate_control_payload(
        "code_seal",
        payload,
        amendment=amendment,
        repo_root=repo_root,
    )
    destination = path_under(artifact_root, CODE_SEAL_RELATIVE, label="V5 code seal")
    _require(not _lexists(destination), "V5 code seal already exists")
    temporary = destination.with_name(f".{destination.name}.stage.{os.getpid()}.{hashlib.sha256(pretty_json_bytes(payload)).hexdigest()[:16]}")
    data = pretty_json_bytes(payload)
    stage_bytes_exclusive(temporary, data)
    observed = publish_staged_create_if_absent(
        temporary,
        destination,
        expected_size=len(data),
        expected_sha256=hashlib.sha256(data).hexdigest(),
    )
    observed["path"] = CODE_SEAL_RELATIVE
    return observed


def code_seal_mode(
    artifact_root: Path,
    repo_root: Path,
    test_evidence_path: Path,
) -> dict[str, Any]:
    install_network_guard()
    amendment = validate_amendment(artifact_root)
    validate_sealer_runtime_identity(amendment)
    _require(not path_under(artifact_root, OUTPUT_ROOT_RELATIVE.rstrip("/"), label="V5 output root").exists(), "code-seal mode requires output root absent")
    for relative, label in (
        (AUTHORIZATION_RELATIVE, "V5 authorization"),
        (REVIEW_RELATIVE, "V5 independent review"),
        (GO_RELATIVE, "V5 independent GO"),
        (POSTRUN_RELATIVE, "V5 independent postrun"),
        (
            "incidents/TARGET_FREE_MULTISEASON_V5_EXECUTION_OR_PUBLICATION_FAILURE_V1.json",
            "future V5 failure incident",
        ),
    ):
        _require(
            not _lexists(path_under(artifact_root, relative, label=label)),
            f"code-seal mode requires {label} absent",
        )
    code = collect_code_identities(repo_root)
    evidence = strict_json_load(test_evidence_path, label="V5 isolated test evidence")
    validate_test_evidence(evidence, repo_root=repo_root, amendment=amendment)
    static = static_self_validate()
    _require(static["models_fit"] == 0 and static["parquet_logical_value_reads"] == 0, "sealer self-validator nonzero forbidden work")
    payload = build_code_seal_payload(
        amendment=amendment,
        code_identities=code,
        test_evidence=evidence,
        created_utc=utc_now_exact(),
        repo_root=repo_root,
    )
    published = publish_code_seal(
        artifact_root,
        payload,
        amendment=amendment,
        repo_root=repo_root,
    )
    return {
        "status": "PASS_V5_CODE_SEAL_PUBLISHED_NO_EXECUTION_AUTHORITY",
        "code_seal": published,
        "models_fit": 0,
        "parquet_logical_value_reads": 0,
        "network_requests": 0,
        "execution_outputs_published": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--independent-go", type=Path)
    parser.add_argument("--postrun-pass", type=Path)
    parser.add_argument("--code-seal-test-evidence", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    actual_execution_argv = (
        getattr(sys, "orig_argv", None) if argv is None else None
    )
    root = lexical_absolute(args.root)
    repo_root = lexical_absolute(args.repo_root)
    if args.code_seal_test_evidence is not None:
        _require(
            args.authorization is None and args.independent_go is None and args.postrun_pass is None,
            "code-seal mode cannot accept execution controls",
        )
        report = code_seal_mode(
            root,
            repo_root,
            lexical_absolute(args.code_seal_test_evidence),
        )
        print(json.dumps(report, ensure_ascii=True, allow_nan=False, sort_keys=True))
        return 0
    _require(
        args.authorization is not None
        and args.independent_go is not None
        and args.postrun_pass is not None,
        "final mode requires authorization, independent GO, and postrun pass",
    )
    install_network_guard()
    validation = validate_authority_chain(
        root,
        lexical_absolute(args.authorization),
        lexical_absolute(args.independent_go),
        lexical_absolute(args.postrun_pass),
        repo_root=repo_root,
        actual_orig_argv=actual_execution_argv,
    )
    report = finalize_validated_authority_chain(root, validation)
    print(json.dumps(report, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only, no-fit post-run auditor for target-free duplicate execution V3.

The module deliberately imports neither NumPy nor PyArrow until the complete
documentary/control and byte-identity boundary has been validated.  It never
opens the logical values of an upstream source parquet.  The only logical
values it may inspect are the runner's seven staged diagnostic outputs and the
two physical aliases required by the published V3 amendment (the parquet
alias is inspected as metadata only).

There is no persisted audit report: a successful CLI invocation emits one
strict JSON payload to stdout and exits zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import platform
import re
import stat as stat_module
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 3
ARTIFACT_TYPE = "TARGET_FREE_DUPLICATE_NO_FIT_POSTRUN_AUDIT_V3"
PASS_STATUS = "PASS_V3_NO_FIT_POSTRUN_AUDIT"
FAIL_STATUS = "FAIL_V3_NO_FIT_POSTRUN_AUDIT"

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = (
    REPO_ROOT
    / "artifacts"
    / "baram2026_ncei_scada_longrun_20260810_v2"
)

AMENDMENT_REL = "prereg/target_free_multiseason_amendment_v3.json"
AMENDMENT_V4_REL = "prereg/target_free_multiseason_amendment_v4.json"
CODE_SEAL_REL = "prereg/target_free_duplicate_code_seal_v3.json"
AUTH_REL = "prereg/target_free_duplicate_execution_authorization_v3.json"
REVIEW_REL = (
    "independent_redteam/"
    "TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_REVIEW_V3.json"
)
GO_REL = (
    "independent_redteam/"
    "TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_GO_V3.json"
)
POSTRUN_REL = (
    "independent_redteam/"
    "TRACK_A_TARGET_FREE_DUPLICATE_POSTRUN_PASS_V3.json"
)

AMENDMENT_SIZE = 102_361
AMENDMENT_SHA256 = (
    "f9d072fc24ab29608efc6483098ada5a8d5e8129e1d1bd478e1d84438a2e9c14"
)
AMENDMENT_CANONICAL_SIZE = 84_601
AMENDMENT_CANONICAL_SHA256 = (
    "ae5c5d05e63169112799c48a8ede52e8ce29319bc5d9e96927175ccbfd9b91d1"
)
AMENDMENT_V4_SIZE = 14_385
AMENDMENT_V4_SHA256 = (
    "9f8abbe878bd10e7b91f78a03f44389a4973c8c1cb7da3c94752487957be7980"
)
AMENDMENT_V4_CANONICAL_SIZE = 12_501
AMENDMENT_V4_CANONICAL_SHA256 = (
    "3ad1f38508b373aac70ee19b4253e91b9104f9527cb77952121757226f35abed"
)
AMENDMENT_V4_CREATED_UTC = "2026-08-11T15:00:30.854000Z"

V4_PARQUET_POINTER = "/output_contract/output_serialization_exact/parquet"
V4_CSV_BOUNDARY_POINTER = (
    "/output_contract/output_serialization_exact/csv_boundary_flags"
)
V4_POINTERS = (V4_PARQUET_POINTER, V4_CSV_BOUNDARY_POINTER)
V4_PARQUET_LITERAL = (
    "PYARROW_22_0_0_PARQUET_VERSION_2_6_ZSTD_LEVEL_3_USE_DICTIONARY_FALSE_"
    "WRITE_STATISTICS_TRUE_DATA_PAGE_VERSION_1_0_ROW_GROUP_SIZE_65536_INDEX_FALSE_"
    "COERCE_TIMESTAMPS_NONE_EFFECTIVE_NATIVE_NS_FOR_VERSION_2_6_"
    "ALLOW_TRUNCATED_TIMESTAMPS_FALSE_USE_DEPRECATED_INT96_FALSE_STORE_SCHEMA_TRUE_"
    "WRITE_PAGE_INDEX_FALSE_FLOAT_COLUMNS_BINARY64"
)
V4_CSV_BOUNDARY_LITERAL = (
    "CSV_BOOLEAN_CELLS_UPPERCASE_TRUE_OR_FALSE;BOUNDARY_EQUALITY_FLAGS_EMPTY_IF_NONE_"
    "ELSE_SEMICOLON_JOINED_ACTIVE_CONTINUOUS_CUTOFF_NAMES_IN_COMPONENT_DECISION_"
    "TRUTH_TABLE_CONTINUOUS_CUTOFFS_EXACT_ORDER_WITH_NO_DUPLICATES"
)
PARQUET_WRITE_KWARGS_EXACT = {
    "version": "2.6",
    "compression": "zstd",
    "compression_level": 3,
    "use_dictionary": False,
    "write_statistics": True,
    "data_page_version": "1.0",
    "row_group_size": 65_536,
    "coerce_timestamps": None,
    "allow_truncated_timestamps": False,
    "use_deprecated_int96_timestamps": False,
    "store_schema": True,
    "write_page_index": False,
}
PYARROW_WRITE_TABLE_UNLISTED_DEFAULTS_EXACT = {
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

CODE_ROLE_PATHS = {
    "runner": "scripts/run_noaa_gfs_target_free_duplicate_v3.py",
    "runner_test": "tests/test_noaa_gfs_target_free_duplicate_v3.py",
    "auditor": "scripts/audit_noaa_gfs_target_free_duplicate_v3.py",
    "auditor_test": "tests/test_audit_noaa_gfs_target_free_duplicate_v3.py",
    "sealer": "scripts/seal_noaa_gfs_target_free_duplicate_v3.py",
    "sealer_test": "tests/test_seal_noaa_gfs_target_free_duplicate_v3.py",
}

OUTPUT_ROOT_REL = "modeling/target_free_multiseason_v3"
TRANSACTION_REL = OUTPUT_ROOT_REL + "/.transaction_v3"
ALIAS_SPECS = {
    "FIELD_CENSUS_LOCK.json": "manifest_census_v1.json",
    "PROVENANCE_LEDGER.parquet": "raw/RAW_RANGE_MANIFEST.parquet",
}
FIELD_CENSUS_KEYS = (
    "access_ledger",
    "artifact_type",
    "created_utc",
    "day_summary",
    "field_range_census",
    "field_summary",
    "labels_read",
    "models_fit",
    "object_census",
    "parent_preregister_manifest",
    "progress_checkpoints",
    "raw_download_plan",
    "raw_downloaded_bytes",
    "raw_network_requests",
    "reproduction_code",
    "schema_version",
    "submission_csv_created",
    "summary",
    "test_code",
)
RUNNER_OUTPUTS = (
    "TARGET_FREE_DUPLICATE_METRICS.csv",
    "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv",
    "TARGET_FREE_FAMILY_DECISION.json",
    "TARGET_FREE_DUPLICATE_OOF_V3.parquet",
    "TARGET_FREE_FIT_LEDGER_V3.json",
    "TARGET_FREE_DISTRIBUTION_STABILITY_V3.json",
    "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3.parquet",
)
SEALER_OUTPUTS = (
    "TARGET_FREE_FAMILY_LOCK.json",
    "TRACK_A_TARGET_FREE_RUN_MANIFEST.json",
)

FILE_IDENTITY_KEYS = frozenset(("path", "size_bytes", "sha256"))
AMENDMENT_IDENTITY_KEYS = frozenset(
    ("path", "size_bytes", "sha256", "canonical_size_bytes", "canonical_sha256")
)
CODE_IDENTITY_KEYS = frozenset(CODE_ROLE_PATHS)
OUTPUT_NAMESPACE = {
    "root": OUTPUT_ROOT_REL + "/",
    "runner_transaction_root": TRANSACTION_REL + "/",
    "require_absent_at_go": True,
    "single_attempt_only": True,
}

CODE_SEAL_TOP_KEYS = frozenset(
    (
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "amendment",
        "amendment_v4",
        "code_identities",
        "test_evidence",
        "output_namespace",
        "authority_scope",
        "required_next_controls",
        "publisher",
    )
)
AUTH_TOP_KEYS = frozenset(
    (
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "attempt_id",
        "amendment",
        "amendment_v4",
        "code_seal",
        "code_identities",
        "output_namespace",
        "execution_budget",
        "required_commands",
        "required_review",
        "required_go",
        "authority_scope",
        "publisher",
    )
)
REVIEW_TOP_KEYS = frozenset(
    (
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "attempt_id",
        "amendment",
        "amendment_v4",
        "code_seal",
        "authorization",
        "code_identities",
        "output_namespace",
        "independent_checks",
        "authority_scope",
        "verdict",
        "publisher",
    )
)
GO_TOP_KEYS = frozenset(
    (
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "attempt_id",
        "amendment",
        "amendment_v4",
        "code_seal",
        "authorization",
        "independent_review",
        "code_identities",
        "output_namespace",
        "required_commands",
        "authority_scope",
        "single_attempt",
        "publisher",
    )
)

POSTRUN_TOP_KEYS = frozenset(
    (
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "attempt_id",
        "amendment",
        "amendment_v4",
        "code_seal",
        "authorization",
        "independent_review",
        "independent_go",
        "code_identities",
        "audit_result",
        "staged_output_identities",
        "authority_scope",
        "verdict",
        "publisher",
    )
)

AUDITOR_STDOUT_TOP_KEYS = frozenset(
    (
        "schema_version",
        "artifact_type",
        "status",
        "attempt_id",
        "amendment",
        "amendment_v4",
        "authorization",
        "independent_go",
        "code_identities",
        "staged_output_identities",
        "output_counts",
        "decision",
        "poststate",
        "files_written",
        "network_requests",
        "models_fit",
        "forbidden_reads",
    )
)

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
AUTHORITY_SCOPE = {
    "single_attempt_authorized": True,
    "alias_source_byte_copy_authorized": True,
    "alias_publication_authorized": True,
    "decoded_predictor_value_read_authorized": True,
    "primary_model_fit_units_authorized": 72,
    "analytic_affine_fit_units_authorized": 1260,
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
REVIEW_CHECKS = {
    key: True
    for key in (
        "amendment_identity",
        "code_seal_identity",
        "authorization_identity",
        "code_identities",
        "test_evidence",
        "bound_inputs",
        "runtime_identity",
        "output_namespace_zero_state",
        "single_attempt_budget",
        "required_commands",
        "authority_scope",
        "chronology",
    )
}
EXECUTION_BUDGET = {
    "attempt_count": 1,
    "alias_count": 2,
    "primary_model_fit_units": 72,
    "analytic_affine_fit_units": 1260,
    "total_decision_fit_units": 1332,
    "fit_ledger_rows": 1332,
    "oof_rows": 352512,
    "metrics_csv_rows": 333,
    "stability_csv_rows": 348,
    "distribution_records": 624,
    "endpoint_pooled_records": 3,
    "derived_rows": 414720,
    "runner_staged_outputs": 7,
    "final_complete_outputs": 11,
}
SINGLE_ATTEMPT = {
    "attempt_number": 1,
    "output_root_was_absent_at_go": True,
    "retry_allowed": False,
    "prior_v3_failure_incident": None,
    "manifest_was_absent_at_go": True,
}
REQUIRED_NEXT_CONTROLS = {
    "authorization": AUTH_REL,
    "independent_review": REVIEW_REL,
    "independent_go": GO_REL,
    "postrun_pass": POSTRUN_REL,
}

OUTPUT_COUNTS = {
    "aliases": 2,
    "runner_staged_outputs": 7,
    "metrics_rows": 333,
    "stability_rows": 348,
    "oof_rows": 352512,
    "fit_ledger_rows": 1332,
    "distribution_records": 624,
    "endpoint_pooled_metric_records": 3,
    "derived_rows": 414720,
}
POSTSTATE = {
    "output_root_entries": 3,
    "transaction_entries": 7,
    "final_publication_files": 0,
    "manifest_present": False,
    "family_lock_present": False,
    "temporary_files": 0,
    "active_locks": 0,
}

# The live auditor runs before the sealer creates either document.  These
# constants and validators freeze the sealer's two documentary payloads
# without making their pre-publication absence an audit failure.
LOCK_TOP_KEYS = frozenset(
    (
        "schema_version",
        "artifact_type",
        "status",
        "selected_family",
        "ordered_raw_columns",
        "ordered_derived_columns",
        "units",
        "grib_selectors",
        "site_group_scope",
        "spatial_transforms",
        "parent_and_independent_gate_results",
        "bound_input_identities",
        "code_config_runtime_identities",
        "postrun_pass_identity",
        "no_label_attestation",
        "downstream_authority",
    )
)
MANIFEST_TOP_KEYS = frozenset(
    (
        "schema_version",
        "artifact_type",
        "status",
        "terminal_state",
        "created_utc",
        "code_config_runtime_identities",
        "control_identities",
        "alias_identities",
        "output_identities",
        "planned_completed_skipped_fit_counts",
        "oof_slot_counts",
        "threadpool_info_before_inside_after",
        "no_forbidden_access_attestation",
        "manifest_is_last_commit_marker",
    )
)
LOCK_ARTIFACT_TYPE = "TARGET_FREE_FAMILY_LOCK_V3"
MANIFEST_ARTIFACT_TYPE = "TRACK_A_TARGET_FREE_RUN_MANIFEST_V3"
MANIFEST_STATUS = "COMMITTED_V3_TARGET_FREE_DECISION"
LOCK_DOWNSTREAM_AUTHORITY = (
    "NONE_REQUIRES_SEPARATE_FULL_EXPANSION_PREREG_CODE_SEAL_AUTH_REVIEW_GO_AND_POSTRUN_PASS"
)
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
CODE_CONFIG_RUNTIME_KEYS = frozenset(
    ("amendment", "amendment_v4", "code_identities", "runtime_identity")
)
CONTROL_IDENTITY_KEYS = frozenset(
    ("code_seal", "authorization", "independent_review", "independent_go", "postrun_pass")
)
FIT_COUNT_TREE_KEYS = frozenset(
    (
        "planned_counts",
        "executed_counts",
        "internal_method_call_counts",
        "skipped_counts",
        "zero_fit_counts",
    )
)
OOF_SLOT_COUNT_KEYS = frozenset(
    (
        "planned_oof_slots",
        "completed_oof_slots",
        "skipped_oof_slots",
        "independent_selected_oof_slots",
    )
)
THREADPOOL_EVIDENCE_KEYS = frozenset(
    (
        "capture_phases",
        "capture_counts",
        "event_count",
        "event_chain_sha256",
        "first_observation_by_phase",
        "last_observation_by_phase",
    )
)
NO_FORBIDDEN_ACCESS_ATTESTATION = {
    "labels_read": False,
    "arrays_2024_2025_read": False,
    "network_requests": 0,
    "generation_model_fit_units": 0,
    "submission_csv_files_written": 0,
}
OUTPUT_IDENTITY_KEYS = frozenset(
    ("path", "size_bytes", "sha256", "format", "row_count", "logical_sha256")
)
FINAL_OUTPUT_ORDER = RUNNER_OUTPUTS + ("TARGET_FREE_FAMILY_LOCK.json",)
FINAL_OUTPUT_FORMATS = {
    "TARGET_FREE_DUPLICATE_METRICS.csv": "CSV",
    "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv": "CSV",
    "TARGET_FREE_FAMILY_DECISION.json": "JSON",
    "TARGET_FREE_DUPLICATE_OOF_V3.parquet": "PARQUET",
    "TARGET_FREE_FIT_LEDGER_V3.json": "JSON",
    "TARGET_FREE_DISTRIBUTION_STABILITY_V3.json": "JSON",
    "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3.parquet": "PARQUET",
    "TARGET_FREE_FAMILY_LOCK.json": "JSON",
}
FINAL_OUTPUT_ROWS = {
    "TARGET_FREE_DUPLICATE_METRICS.csv": 333,
    "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv": 348,
    "TARGET_FREE_FAMILY_DECISION.json": 1,
    "TARGET_FREE_DUPLICATE_OOF_V3.parquet": 352_512,
    "TARGET_FREE_FIT_LEDGER_V3.json": 1_332,
    "TARGET_FREE_DISTRIBUTION_STABILITY_V3.json": 624,
    "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3.parquet": 414_720,
    "TARGET_FREE_FAMILY_LOCK.json": 1,
}
RAW_COLUMN_UNITS = {
    "HPBL_surface": "m",
    "UGRD_925mb": "m/s",
    "VGRD_925mb": "m/s",
    "UGRD_950mb": "m/s",
    "VGRD_950mb": "m/s",
    "UGRD_975mb": "m/s",
    "VGRD_975mb": "m/s",
    "UGRD_1000mb": "m/s",
    "VGRD_1000mb": "m/s",
}
GRIB_SELECTOR_BY_RAW_COLUMN = {
    "HPBL_surface": {"variable": "HPBL", "level": "surface"},
    "UGRD_925mb": {"variable": "UGRD", "level": "925 mb"},
    "VGRD_925mb": {"variable": "VGRD", "level": "925 mb"},
    "UGRD_950mb": {"variable": "UGRD", "level": "950 mb"},
    "VGRD_950mb": {"variable": "VGRD", "level": "950 mb"},
    "UGRD_975mb": {"variable": "UGRD", "level": "975 mb"},
    "VGRD_975mb": {"variable": "VGRD", "level": "975 mb"},
    "UGRD_1000mb": {"variable": "UGRD", "level": "1000 mb"},
    "VGRD_1000mb": {"variable": "VGRD", "level": "1000 mb"},
}

ESTIMATOR_ORDER = ("RIDGE_PIPELINE", "EXTRA_TREES")
RAW_COMPONENT_RESULT_KEYS = (
    "response",
    "classification",
    "counts_as_nonredundant",
    "independent_estimator",
    "parent_reference_estimators",
    "parent_redundancy_boolean",
    "affine_qualifying_predictors",
    "cell_stability_pass",
    "distribution_stability_pass",
    "boundary_equality_flags",
    "ambiguity_reason",
)
ENDPOINT_VETO_RESULT_KEYS = (
    "diagnostic",
    "verdict",
    "source_component_estimator_binding",
    "pooled_stable_margin_pass",
    "cell_stability_pass",
    "distribution_stability_pass",
    "boundary_equality_flags",
    "clear_pass",
    "ambiguity_reason",
)
THRESHOLD_WITNESS_KEYS = (
    "cutoff_name",
    "scope",
    "diagnostic",
    "value",
    "value_float_hex",
    "cutoff_float_hex",
    "boundary_equal",
)
TIED_WITNESS_KEYS = (
    "tie_kind",
    "diagnostic",
    "estimators",
    "metric_name",
    "metric_values_float_hex",
    "resolution",
    "global_ambiguity",
)
RAW_CLASSIFICATIONS = (
    "CLEAR_PHYSICAL_OR_COVERAGE_FAILURE",
    "CLEAR_NONNOVEL_CONSTANT",
    "AMBIGUOUS_THRESHOLD_EQUALITY",
    "CLEAR_REDUNDANT_AFFINE",
    "AMBIGUOUS_PARENT_ESTIMATOR_TIE",
    "CLEAR_REDUNDANT_PARENT_OOF",
    "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE",
    "CLEAR_REDUNDANT_INDEPENDENT_EXACT_DUPLICATE",
    "PASS_NONREDUNDANT_MARGIN",
    "CLEAR_UNSTABLE_FAILURE",
    "AMBIGUOUS_GRAY_ZONE",
)
PREMODEL_CLASSIFICATIONS = frozenset(
    ("CLEAR_PHYSICAL_OR_COVERAGE_FAILURE", "CLEAR_NONNOVEL_CONSTANT")
)
FIT_SKIP_REASONS = frozenset(
    (
        "SKIPPED_PREMODEL_CLEAR_PHYSICAL_WIND_FAMILY_VETO",
        "SKIPPED_PREMODEL_CLEAR_PHYSICAL_PBL_FAMILY_VETO",
        "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3",
        "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_POOLED_SST_LTE_1E_MINUS_12",
    )
)
AMBIGUOUS_CLASSIFICATIONS = frozenset(
    (
        "AMBIGUOUS_THRESHOLD_EQUALITY",
        "AMBIGUOUS_PARENT_ESTIMATOR_TIE",
        "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE",
        "AMBIGUOUS_GRAY_ZONE",
    )
)
ENDPOINT_VERDICTS = frozenset(
    (
        "PASS_ENDPOINT_NONREDUNDANT_MARGIN",
        "CLEAR_ENDPOINT_VETO_FAILURE",
        "AMBIGUOUS_THRESHOLD_EQUALITY",
        "AMBIGUOUS_RAW_ESTIMATOR_TIE",
    )
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
CUTOFF_NAMES = tuple(CUTOFF_VALUES)
TIE_KINDS = ("INDEPENDENT_RMSE", "PARENT_MAX_R2")
TIE_RESOLUTIONS = frozenset(
    (
        "GLOBAL_AMBIGUITY",
        "SHARED_PARENT_REDUNDANCY_TRUE",
        "SHARED_PARENT_REDUNDANCY_FALSE",
        "MASKED_BY_PRIOR_AFFINE_REDUNDANCY",
    )
)
FIXED_FAMILY_PRIORITY = ("LOW_LEVEL_ISOBARIC_WIND_PROFILE", "PBL_HEIGHT")
WIND_RESPONSE_RE = re.compile(r"^[UV]GRD_(925|950|975|1000)mb$")

RFC3339_MICRO_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ATTEMPT_ID_RE = re.compile(r"^target_free_duplicate_v3__[0-9]{8}T[0-9]{12}Z$")
PYTEST_SUMMARY_RE = re.compile(
    r"^(?P<passed>[0-9]+) passed"
    r"(?:, (?P<skipped>[0-9]+) skipped)?"
    r"(?:, (?P<deselected>[0-9]+) deselected)? "
    r"in (?P<seconds>[0-9]+(?:\.[0-9]+)?)s"
    r"(?: \((?P<hours>0|[1-9][0-9]*):(?P<minutes>[0-5][0-9]):(?P<clock_seconds>[0-5][0-9])\))?$"
)


class AuditError(RuntimeError):
    """Fail-closed validation error."""


@dataclass(frozen=True)
class PredataContext:
    root: Path
    repo: Path
    amendment: dict[str, Any]
    amendment_identity: dict[str, Any]
    amendment_v4: dict[str, Any]
    amendment_v4_identity: dict[str, Any]
    code_seal: dict[str, Any]
    code_seal_identity: dict[str, Any]
    authorization: dict[str, Any]
    authorization_identity: dict[str, Any]
    review: dict[str, Any]
    review_identity: dict[str, Any]
    independent_go: dict[str, Any]
    independent_go_identity: dict[str, Any]
    attempt_id: str


def _fail(message: str) -> None:
    raise AuditError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _json_type_exact_equal(observed: Any, expected: Any) -> bool:
    """Compare JSON-compatible trees without Python bool/int/float aliasing."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return observed.keys() == expected.keys() and all(
            _json_type_exact_equal(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _json_type_exact_equal(left, right)
            for left, right in zip(observed, expected)
        )
    return observed == expected


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    _require(not raw.startswith(b"\xef\xbb\xbf"), f"{label}: UTF-8 BOM forbidden")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AuditError(f"{label}: invalid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label}: invalid strict JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{label}: top-level object required")
    return value


def strict_json_file(path: Path, *, label: str | None = None) -> dict[str, Any]:
    path = _reject_linklike_components(path, label=label or str(path))
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AuditError(f"{label or path}: unreadable") from exc
    return strict_json_bytes(raw, label=label or str(path))


def strict_pretty_control_json_file(
    path: Path, *, label: str | None = None
) -> dict[str, Any]:
    """Load an immutable control and bind its exact frozen physical bytes."""

    resolved_label = label or str(path)
    path = _reject_linklike_components(path, label=resolved_label)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AuditError(f"{resolved_label}: unreadable") from exc
    value = strict_json_bytes(raw, label=resolved_label)
    _require(
        raw == pretty_json_bytes(value, ensure_ascii=True),
        f"{resolved_label}: physical pretty JSON serialization mismatch",
    )
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: Any, *, ensure_ascii: bool = True) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute normalized path without following any link."""
    return Path(os.path.abspath(os.fspath(path)))


def _reject_linklike_components(path: Path, *, label: str) -> Path:
    """Reject symlink, junction, or other reparse components before a read."""
    absolute = _lexical_absolute(path)
    anchor = Path(absolute.anchor)
    components = [anchor]
    cursor = anchor
    for part in absolute.parts[1:]:
        cursor = cursor / part
        components.append(cursor)
    reparse_flag = int(getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    for component in components:
        try:
            if component.is_symlink():
                _fail(f"{label}: symlink component forbidden: {component}")
            is_junction = getattr(component, "is_junction", None)
            if callable(is_junction) and is_junction():
                _fail(f"{label}: junction component forbidden: {component}")
            metadata = component.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise AuditError(f"{label}: cannot inspect path component: {component}") from exc
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if attributes & reparse_flag:
            _fail(f"{label}: reparse component forbidden: {component}")
    return absolute


def _same_lexical_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.fspath(_lexical_absolute(left))) == os.path.normcase(
        os.fspath(_lexical_absolute(right))
    )


def sha256_file(path: Path) -> tuple[int, str]:
    path = _reject_linklike_components(path, label=f"hash path {path}")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
    except OSError as exc:
        raise AuditError(f"cannot hash {path}") from exc
    return size, digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    actual = frozenset(value)
    frozen = frozenset(expected)
    if actual != frozen:
        _fail(
            f"{label}: exact keys mismatch; missing={sorted(frozen-actual)!r} "
            f"extra={sorted(actual-frozen)!r}"
        )


def _path_text(path: Path) -> str:
    return _reject_linklike_components(path, label=f"identity path {path}").as_posix()


def _artifact_path(root: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and relative != "", "empty artifact path")
    candidate = Path(relative)
    _require(not candidate.is_absolute(), f"artifact path must be relative: {relative}")
    _require(".." not in candidate.parts, f"artifact path traversal: {relative}")
    lexical_root = _reject_linklike_components(root, label="artifact root")
    lexical_candidate = _lexical_absolute(lexical_root / candidate)
    try:
        common = os.path.commonpath((os.fspath(lexical_root), os.fspath(lexical_candidate)))
    except ValueError as exc:
        raise AuditError(f"artifact path escapes root: {relative}") from exc
    _require(
        os.path.normcase(common) == os.path.normcase(os.fspath(lexical_root)),
        f"artifact path escapes root: {relative}",
    )
    return _reject_linklike_components(lexical_candidate, label=f"artifact path {relative}")


def file_identity(path: Path, bound_path: str) -> dict[str, Any]:
    size, digest = sha256_file(path)
    return {"path": bound_path, "size_bytes": size, "sha256": digest}


def _validate_identity_record(record: Any, label: str) -> dict[str, Any]:
    _require(isinstance(record, dict), f"{label}: identity object required")
    _exact_keys(record, FILE_IDENTITY_KEYS, label)
    _require(isinstance(record["path"], str) and record["path"] != "", f"{label}: path invalid")
    _require(type(record["size_bytes"]) is int and record["size_bytes"] >= 0, f"{label}: size invalid")
    _require(isinstance(record["sha256"], str) and SHA256_RE.fullmatch(record["sha256"]) is not None, f"{label}: sha256 invalid")
    return dict(record)


def verify_file_identity(
    record: Mapping[str, Any],
    path: Path,
    *,
    label: str,
    expected_path: str | None = None,
) -> dict[str, Any]:
    _exact_keys(record, FILE_IDENTITY_KEYS, label)
    if expected_path is not None:
        _require(record["path"] == expected_path, f"{label}: path mismatch")
    _require(
        isinstance(record["size_bytes"], int)
        and not isinstance(record["size_bytes"], bool)
        and record["size_bytes"] >= 0,
        f"{label}: invalid size",
    )
    _require(
        isinstance(record["sha256"], str) and SHA256_RE.fullmatch(record["sha256"]),
        f"{label}: invalid sha256",
    )
    actual = file_identity(path, str(record["path"]))
    _require(actual == dict(record), f"{label}: physical identity mismatch")
    return actual


def _parse_utc_microseconds(value: Any, label: str) -> int:
    _require(isinstance(value, str) and RFC3339_MICRO_RE.fullmatch(value), f"{label}: invalid RFC3339 microsecond UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise AuditError(f"{label}: invalid UTC timestamp") from exc
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = parsed - epoch
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _validate_attempt_id(value: Any) -> str:
    _require(isinstance(value, str) and ATTEMPT_ID_RE.fullmatch(value), "attempt_id syntax invalid")
    suffix = value.removeprefix("target_free_duplicate_v3__")
    try:
        parsed = datetime.strptime(suffix, "%Y%m%dT%H%M%S%fZ")
    except ValueError as exc:
        raise AuditError("attempt_id calendar timestamp invalid") from exc
    _require(parsed.strftime("%Y%m%dT%H%M%S%fZ") == suffix, "attempt_id timestamp roundtrip mismatch")
    return value


def _validate_amendment(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _artifact_path(root, AMENDMENT_REL)
    size, digest = sha256_file(path)
    _require(size == AMENDMENT_SIZE, "V3 amendment physical size mismatch")
    _require(digest == AMENDMENT_SHA256, "V3 amendment physical sha256 mismatch")
    raw = path.read_bytes()
    amendment = strict_json_bytes(raw, label=AMENDMENT_REL)
    compact = canonical_json_bytes(amendment)
    _require(len(compact) == AMENDMENT_CANONICAL_SIZE, "V3 amendment canonical size mismatch")
    _require(sha256_bytes(compact) == AMENDMENT_CANONICAL_SHA256, "V3 amendment canonical sha256 mismatch")
    _require(pretty_json_bytes(amendment, ensure_ascii=False) == raw, "V3 amendment pretty serialization mismatch")
    strict = amendment.get("strict_schema")
    _require(isinstance(strict, dict), "V3 amendment strict_schema missing")
    _exact_keys(amendment, strict.get("top_level_exact_keys", ()), "V3 amendment")
    _require(len(amendment) == strict.get("top_level_exact_key_count") == 26, "V3 amendment top-level count mismatch")
    _require(amendment.get("schema_version") == 3, "V3 amendment schema version mismatch")
    _require(
        amendment.get("artifact_type")
        == "TARGET_FREE_MULTISEASON_PREREGISTRATION_APPEND_ONLY_EXECUTION_AMENDMENT",
        "V3 amendment artifact type mismatch",
    )
    _parse_utc_microseconds(amendment.get("created_utc"), "amendment.created_utc")
    identity = {
        "path": AMENDMENT_REL,
        "size_bytes": size,
        "sha256": digest,
        "canonical_size_bytes": len(compact),
        "canonical_sha256": sha256_bytes(compact),
    }
    return amendment, identity


def _validate_amendment_v4(
    root: Path,
    amendment: Mapping[str, Any],
    amendment_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the append-only V4 two-pointer serialization correction."""

    path = _artifact_path(root, AMENDMENT_V4_REL)
    size, digest = sha256_file(path)
    _require(size == AMENDMENT_V4_SIZE, "V4 amendment physical size mismatch")
    _require(digest == AMENDMENT_V4_SHA256, "V4 amendment physical sha256 mismatch")
    raw = path.read_bytes()
    value = strict_json_bytes(raw, label=AMENDMENT_V4_REL)
    compact = canonical_json_bytes(value)
    _require(len(compact) == AMENDMENT_V4_CANONICAL_SIZE, "V4 amendment canonical size mismatch")
    _require(
        sha256_bytes(compact) == AMENDMENT_V4_CANONICAL_SHA256,
        "V4 amendment canonical sha256 mismatch",
    )
    _require(
        pretty_json_bytes(value, ensure_ascii=False) == raw,
        "V4 amendment pretty serialization mismatch",
    )
    strict = value.get("strict_schema")
    _require(isinstance(strict, dict), "V4 amendment strict_schema missing")
    _exact_keys(value, strict.get("top_level_exact_keys", ()), "V4 amendment")
    _require(
        len(value) == strict.get("top_level_exact_key_count") == 18,
        "V4 amendment top-level count mismatch",
    )
    for key, expected_count in strict.get("top_level_object_key_counts", {}).items():
        _require(
            key in value
            and isinstance(value[key], dict)
            and len(value[key]) == expected_count,
            f"V4 amendment nested count mismatch: {key}",
        )
    _require(value.get("schema_version") == 4, "V4 amendment schema version mismatch")
    _require(
        value.get("artifact_type")
        == "TARGET_FREE_MULTISEASON_PREREGISTRATION_APPEND_ONLY_DOCUMENTARY_CORRECTION_V4",
        "V4 amendment artifact type mismatch",
    )
    _require(
        value.get("status")
        == "PASS_DOCUMENTARY_V4_PARQUET_API_AND_CSV_BOUNDARY_SERIALIZATION_CORRECTION_ONLY_NO_EXECUTION_AUTHORITY",
        "V4 amendment status mismatch",
    )
    _require(value.get("canonical_path") == AMENDMENT_V4_REL, "V4 canonical path mismatch")
    _require(value.get("created_utc") == AMENDMENT_V4_CREATED_UTC, "V4 created_utc drift")
    _parse_utc_microseconds(value["created_utc"], "amendment_v4.created_utc")
    _require(value.get("bound_v3_amendment") == amendment_identity, "V4 bound V3 identity mismatch")

    correction = value.get("correction_scope")
    _require(isinstance(correction, dict), "V4 correction_scope missing")
    _require(correction.get("correction_count_exact") == 2, "V4 correction count mismatch")
    _require(
        correction.get("superseded_pointers_exact_order") == list(V4_POINTERS),
        "V4 correction pointer order mismatch",
    )
    serialization = amendment["output_contract"]["output_serialization_exact"]
    expected_old = {
        V4_PARQUET_POINTER: serialization["parquet"],
        V4_CSV_BOUNDARY_POINTER: serialization["csv_boundary_flags"],
    }
    expected_new = {
        V4_PARQUET_POINTER: V4_PARQUET_LITERAL,
        V4_CSV_BOUNDARY_POINTER: V4_CSV_BOUNDARY_LITERAL,
    }
    _require(correction.get("v3_literals_exact") == expected_old, "V4 old literals mismatch")
    _require(correction.get("replacement_literals_exact") == expected_new, "V4 replacement literals mismatch")
    _require(
        correction.get("all_other_v3_parsed_semantic_pointers_unchanged") is True
        and correction.get("scope_expansion_forbidden") is True,
        "V4 correction closure mismatch",
    )

    parquet = value.get("corrected_parquet_serialization")
    _require(isinstance(parquet, dict), "V4 parquet correction missing")
    _require(parquet.get("corrected_v3_literal") == V4_PARQUET_LITERAL, "V4 parquet literal mismatch")
    _require(
        parquet.get("explicit_write_table_kwargs_exact") == PARQUET_WRITE_KWARGS_EXACT,
        "V4 exact12 parquet kwargs mismatch",
    )
    _require(len(parquet["explicit_write_table_kwargs_exact"]) == 12, "V4 parquet kwargs not exact12")
    invariants = parquet.get("unchanged_non_kwarg_invariants_exact")
    _require(isinstance(invariants, dict) and len(invariants) == 7, "V4 parquet invariants not exact7")
    _require(
        invariants.get("all_unlisted_write_table_kwargs_use_pyarrow_22_0_0_runtime_defaults") is True
        and invariants.get("unlisted_write_table_kwargs_override_forbidden") is True
        and invariants.get("pyarrow_runtime_version_exact") == "22.0.0",
        "V4 unlisted PyArrow kwargs closure mismatch",
    )
    native = parquet.get("native_ns_validation_exact")
    _require(isinstance(native, dict) and len(native) == 8, "V4 native-ns validation not exact8")
    _require(
        native.get("valid_time_utc_arrow_type_exact") == "timestamp[ns, tz=UTC]"
        and native.get("target_operating_day_kst_arrow_type_exact") == "date32[day]"
        and native.get("both_columns_nonnullable") is True
        and native.get("nanosecond_residue_witness_exact") == 123,
        "V4 native-ns schema contract mismatch",
    )

    csv_contract = value.get("corrected_csv_boundary_serialization")
    _require(isinstance(csv_contract, dict) and len(csv_contract) == 6, "V4 CSV correction not exact6")
    _require(csv_contract.get("corrected_v3_literal") == V4_CSV_BOUNDARY_LITERAL, "V4 CSV literal mismatch")
    _require(
        csv_contract.get("csv_boolean_cells_exact_values") == ["TRUE", "FALSE"]
        and csv_contract.get("boundary_equality_flags_none_representation") == "EMPTY_FIELD"
        and csv_contract.get("boundary_equality_flags_present_representation")
        == "SEMICOLON_JOINED_ACTIVE_CONTINUOUS_CUTOFF_NAMES"
        and csv_contract.get("boundary_equality_flag_order_source")
        == "component_decision_truth_table.continuous_cutoffs_exact"
        and csv_contract.get("duplicate_or_inactive_cutoff_names_forbidden") is True,
        "V4 CSV boundary grammar mismatch",
    )

    affected = value.get("affected_output_schemas_exact")
    _exact_keys(
        affected,
        ("TARGET_FREE_DUPLICATE_OOF_V3.parquet", "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3.parquet"),
        "V4 affected output schemas",
    )
    expected_columns = {
        "valid_time_utc": {"arrow_type": "timestamp[ns, tz=UTC]", "nullable": False},
        "target_operating_day_kst": {"arrow_type": "date32[day]", "nullable": False},
    }
    for basename, contract in affected.items():
        _require(contract == expected_columns, f"V4 affected schema mismatch: {basename}")

    preserved = value.get("preserved_v3_semantics")
    _require(isinstance(preserved, dict), "V4 preserved semantics missing")
    _require(
        preserved.get("parsed_semantic_pointer_diff_count_exact") == 2
        and preserved.get("parsed_semantic_pointer_diff_paths_exact") == list(V4_POINTERS)
        and all(
            item is True
            for key, item in preserved.items()
            if key not in {"parsed_semantic_pointer_diff_count_exact", "parsed_semantic_pointer_diff_paths_exact"}
        ),
        "V4 preserved-semantics closure mismatch",
    )

    authority = value.get("authority_scope")
    _require(isinstance(authority, dict) and len(authority) == 16, "V4 authority exact16 required")
    _require(
        authority.get("authorized_code_and_test_paths") == list(CODE_ROLE_PATHS.values()),
        "V4 authorized six paths mismatch",
    )
    _require(
        authority.get("code_and_test_implementation_authorized") is True
        and authority.get("documentary_correction") is True,
        "V4 documentary implementation authority mismatch",
    )
    _require(
        all(
            item is False
            for key, item in authority.items()
            if key
            not in {
                "authorized_code_and_test_paths",
                "code_and_test_implementation_authorized",
                "documentary_correction",
            }
        ),
        "V4 forbidden authority became true",
    )

    bindings = value.get("required_future_control_bindings")
    _require(isinstance(bindings, dict) and len(bindings) == 16, "V4 future control bindings not exact16")
    _require(
        bindings.get("v3_identity_field_name") == "amendment"
        and bindings.get("v4_identity_field_name") == "amendment_v4"
        and bindings.get("identity_object_exact_keys")
        == ["path", "size_bytes", "sha256", "canonical_size_bytes", "canonical_sha256"]
        and bindings.get("code_seal_top_level_key_count") == 12
        and bindings.get("execution_authorization_top_level_key_count") == 16
        and bindings.get("independent_review_top_level_key_count") == 15
        and bindings.get("independent_go_top_level_key_count") == 16
        and bindings.get("independent_postrun_top_level_key_count") == 17
        and bindings.get("auditor_stdout_top_level_key_count") == 17
        and bindings.get("runner_stdout_top_level_key_count") == 11
        and bindings.get("runner_stdout_amendment_v4_field_forbidden") is True,
        "V4 control/stdout binding counts mismatch",
    )
    final_binding = value.get("final_lock_manifest_binding_supersession")
    _require(isinstance(final_binding, dict) and len(final_binding) == 7, "V4 final binding not exact7")
    _require(
        final_binding.get("prior_exact_key_count") == 3
        and final_binding.get("superseding_exact_key_count") == 4
        and final_binding.get("superseding_exact_keys")
        == ["amendment", "amendment_v4", "code_identities", "runtime_identity"]
        and final_binding.get("required_in_both_final_lock_and_manifest") is True,
        "V4 final lock/manifest exact4 mismatch",
    )

    identity = {
        "path": AMENDMENT_V4_REL,
        "size_bytes": size,
        "sha256": digest,
        "canonical_size_bytes": len(compact),
        "canonical_sha256": sha256_bytes(compact),
    }
    return value, identity


def _resolve_bound_input(root: Path, path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return _reject_linklike_components(candidate, label=f"bound input {path_text}")
    return _artifact_path(root, path_text)


def _validate_bound_inputs(root: Path, amendment: Mapping[str, Any]) -> None:
    bound = amendment.get("bound_inputs")
    _require(isinstance(bound, dict) and len(bound) == 31, "bound_inputs exact31 required")
    for role, record in bound.items():
        _require(isinstance(record, dict), f"bound_inputs.{role}: object required")
        _require(FILE_IDENTITY_KEYS.issubset(record), f"bound_inputs.{role}: identity fields missing")
        verify_file_identity(
            {key: record[key] for key in FILE_IDENTITY_KEYS},
            _resolve_bound_input(root, record["path"]),
            label=f"bound_inputs.{role}",
            expected_path=record["path"],
        )


def _validate_runtime_identity(amendment: Mapping[str, Any]) -> None:
    runtime = amendment.get("runtime_identity")
    _require(isinstance(runtime, dict), "runtime_identity missing")
    _require(platform.python_version() == runtime["python_version"], "Python version drift")
    _require(sys.version == runtime["python_full_version"], "Python full version drift")
    _require(_path_text(Path(sys.executable)) == runtime["python_executable"], "Python executable drift")
    _require(platform.platform() == runtime["platform"], "platform drift")
    for package, expected in runtime["packages"].items():
        distribution = "scikit-learn" if package == "scikit_learn" else package
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise AuditError(f"runtime package missing: {distribution}") from exc
        _require(observed == expected, f"runtime package drift: {distribution}")
    for role, record in runtime["native_libraries_exact"].items():
        path = Path(record["path"])
        size, digest = sha256_file(path)
        _require(size == record["size_bytes"], f"native library size drift: {role}")
        _require(digest == record["sha256"], f"native library sha drift: {role}")


def _expected_commands(root: Path, amendment: Mapping[str, Any]) -> dict[str, list[str]]:
    python = str(Path(amendment["runtime_identity"]["python_executable"]))
    root_text = str(_reject_linklike_components(root, label="command root"))
    auth_text = str(_artifact_path(root, AUTH_REL))
    go_text = str(_artifact_path(root, GO_REL))
    base = ["--root", root_text, "--authorization", auth_text, "--independent-go", go_text]
    return {
        "runner": [python, "-B", "-m", "scripts.run_noaa_gfs_target_free_duplicate_v3", *base],
        "auditor": [python, "-B", "-m", "scripts.audit_noaa_gfs_target_free_duplicate_v3", *base],
        "sealer": [
            python,
            "-B",
            "-m",
            "scripts.seal_noaa_gfs_target_free_duplicate_v3",
            *base,
            "--postrun-pass",
            str(_artifact_path(root, POSTRUN_REL)),
        ],
    }


def _expected_test_commands(repo: Path, amendment: Mapping[str, Any]) -> dict[str, list[str]]:
    python = str(Path(amendment["runtime_identity"]["python_executable"]))
    result: dict[str, list[str]] = {}
    for role in ("runner", "auditor", "sealer"):
        test_role = role + "_test"
        result[role] = [
            python,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(_reject_linklike_components(repo / CODE_ROLE_PATHS[test_role], label=f"test command {role}")),
        ]
    return result


def _validate_code_identities(
    records: Mapping[str, Any], repo: Path, *, label: str
) -> dict[str, Any]:
    _exact_keys(records, CODE_IDENTITY_KEYS, label)
    validated: dict[str, Any] = {}
    for role, relative in CODE_ROLE_PATHS.items():
        record = records[role]
        _require(isinstance(record, dict), f"{label}.{role}: object required")
        expected_path = _path_text(repo / relative)
        validated[role] = verify_file_identity(
            record,
            repo / relative,
            label=f"{label}.{role}",
            expected_path=expected_path,
        )
    return validated


def _validate_test_evidence(
    evidence: Mapping[str, Any], repo: Path, amendment: Mapping[str, Any]
) -> None:
    _exact_keys(
        evidence,
        (
            "commands",
            "results",
            "all_exit_codes_zero",
            "all_expected_test_files_covered",
            "network_denied",
            "pycache_disabled",
            "code_identities_revalidated",
            "completed_utc",
        ),
        "code_seal.test_evidence",
    )
    expected_commands = _expected_test_commands(repo, amendment)
    _require(
        _json_type_exact_equal(evidence["commands"], expected_commands),
        "test evidence commands mismatch",
    )
    results = evidence["results"]
    _exact_keys(results, ("runner", "auditor", "sealer"), "test_evidence.results")
    for role, result in results.items():
        _exact_keys(
            result,
            ("test_file", "exit_code", "passed", "skipped", "deselected", "summary"),
            f"test_evidence.results.{role}",
        )
        _require(result["test_file"] == CODE_ROLE_PATHS[role + "_test"], f"{role} test file mismatch")
        _require(
            isinstance(result["exit_code"], int)
            and not isinstance(result["exit_code"], bool)
            and result["exit_code"] == 0,
            f"{role} test exit code must be integer zero",
        )
        for count_name in ("passed", "skipped", "deselected"):
            count = result[count_name]
            _require(isinstance(count, int) and not isinstance(count, bool) and count >= 0, f"{role} invalid {count_name}")
        _require(result["passed"] > 0, f"{role} test suite has no passes")
        summary = result["summary"]
        match = PYTEST_SUMMARY_RE.fullmatch(summary) if isinstance(summary, str) else None
        _require(match is not None, f"{role} invalid pytest summary")
        assert match is not None
        parsed_counts = {
            "passed": int(match.group("passed")),
            "skipped": int(match.group("skipped") or 0),
            "deselected": int(match.group("deselected") or 0),
        }
        _require(
            all(result[name] == value for name, value in parsed_counts.items()),
            f"{role} pytest summary/count mismatch",
        )
        has_clock = match.group("hours") is not None
        if has_clock:
            clock_seconds = (
                int(match.group("hours")) * 3_600
                + int(match.group("minutes")) * 60
                + int(match.group("clock_seconds"))
            )
            _require(clock_seconds >= 60, f"{role} pytest duration clock below long-duration threshold")
            seconds_text = match.group("seconds")
            whole_text, separator, fraction_text = seconds_text.partition(".")
            displayed_whole = int(whole_text)
            rounded_integer_edge = (
                separator == "."
                and fraction_text != ""
                and set(fraction_text) == {"0"}
                and clock_seconds == displayed_whole - 1
            )
            _require(
                clock_seconds == displayed_whole or rounded_integer_edge,
                f"{role} pytest duration clock mismatch",
            )
    for name in (
        "all_exit_codes_zero",
        "all_expected_test_files_covered",
        "network_denied",
        "pycache_disabled",
        "code_identities_revalidated",
    ):
        _require(evidence[name] is True, f"test_evidence.{name} must be true")
    _parse_utc_microseconds(evidence["completed_utc"], "test_evidence.completed_utc")


def _control_identity(root: Path, relative: str) -> dict[str, Any]:
    return file_identity(_artifact_path(root, relative), relative)


def _validate_code_seal(
    root: Path,
    repo: Path,
    amendment: Mapping[str, Any],
    amendment_identity: Mapping[str, Any],
    amendment_v4_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _artifact_path(root, CODE_SEAL_REL)
    seal = strict_pretty_control_json_file(path, label=CODE_SEAL_REL)
    _exact_keys(seal, CODE_SEAL_TOP_KEYS, "code seal")
    _require(type(seal["schema_version"]) is int and seal["schema_version"] == 3, "code seal schema mismatch")
    _require(seal["artifact_type"] == "TARGET_FREE_DUPLICATE_CODE_SEAL_V3", "code seal artifact mismatch")
    _require(seal["status"] == "SEALED_V3_CODE_AND_TESTS_NO_EXECUTION_AUTHORITY", "code seal status mismatch")
    _require(seal["publisher"] == "V3_SEALER_CODE_SEAL_MODE_CREATE_IF_ABSENT_ONLY", "code seal publisher mismatch")
    _require(_json_type_exact_equal(seal["amendment"], amendment_identity), "code seal amendment crosslink mismatch")
    _require(
        _json_type_exact_equal(seal["amendment_v4"], amendment_v4_identity),
        "code seal V4 amendment crosslink mismatch",
    )
    _require(_json_type_exact_equal(seal["output_namespace"], OUTPUT_NAMESPACE), "code seal output namespace mismatch")
    _require(_json_type_exact_equal(seal["authority_scope"], DOCUMENTARY_SCOPE), "code seal authority scope mismatch")
    _require(
        _json_type_exact_equal(seal["required_next_controls"], REQUIRED_NEXT_CONTROLS),
        "code seal next controls mismatch",
    )
    _validate_code_identities(seal["code_identities"], repo, label="code_seal.code_identities")
    _validate_test_evidence(seal["test_evidence"], repo, amendment)
    created = _parse_utc_microseconds(seal["created_utc"], "code_seal.created_utc")
    completed = _parse_utc_microseconds(seal["test_evidence"]["completed_utc"], "test_evidence.completed_utc")
    amendment_v4_created = _parse_utc_microseconds(AMENDMENT_V4_CREATED_UTC, "amendment_v4.created_utc")
    _require(
        amendment_v4_created < completed <= created,
        "test evidence chronology must be amendment V4 < completed <= code seal",
    )
    return seal, _control_identity(root, CODE_SEAL_REL)


def _validate_authorization(
    root: Path,
    amendment: Mapping[str, Any],
    amendment_identity: Mapping[str, Any],
    amendment_v4_identity: Mapping[str, Any],
    code_seal: Mapping[str, Any],
    code_seal_identity: Mapping[str, Any],
    authorization_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authorization_path = _reject_linklike_components(authorization_path, label="authorization path")
    _require(_same_lexical_path(authorization_path, _artifact_path(root, AUTH_REL)), "authorization path mismatch")
    auth = strict_pretty_control_json_file(authorization_path, label=AUTH_REL)
    _exact_keys(auth, AUTH_TOP_KEYS, "authorization")
    _require(type(auth["schema_version"]) is int and auth["schema_version"] == 3, "authorization schema mismatch")
    _require(auth["artifact_type"] == "TARGET_FREE_DUPLICATE_EXECUTION_AUTHORIZATION_V3", "authorization artifact mismatch")
    _require(auth["status"] == "AUTHORIZED_V3_SINGLE_ATTEMPT_PENDING_INDEPENDENT_REVIEW_AND_GO", "authorization status mismatch")
    _require(auth["publisher"] == "ROOT_AUTHORITY_APPEND_ONLY_ONLY", "authorization publisher mismatch")
    _require(_json_type_exact_equal(auth["amendment"], amendment_identity), "authorization amendment mismatch")
    _require(
        _json_type_exact_equal(auth["amendment_v4"], amendment_v4_identity),
        "authorization V4 amendment mismatch",
    )
    _require(_json_type_exact_equal(auth["code_seal"], code_seal_identity), "authorization code-seal mismatch")
    _require(
        _json_type_exact_equal(auth["code_identities"], code_seal["code_identities"]),
        "authorization code identities mismatch",
    )
    _require(_json_type_exact_equal(auth["output_namespace"], OUTPUT_NAMESPACE), "authorization output namespace mismatch")
    _require(_json_type_exact_equal(auth["execution_budget"], EXECUTION_BUDGET), "authorization execution budget mismatch")
    _require(
        _json_type_exact_equal(auth["required_commands"], _expected_commands(root, amendment)),
        "authorization command mismatch",
    )
    _require(
        _json_type_exact_equal(
            auth["required_review"],
            {"path": REVIEW_REL, "status_required": "PASS_V3_EXECUTION_REVIEW_PENDING_INDEPENDENT_GO"},
        ),
        "authorization required_review mismatch",
    )
    _require(
        _json_type_exact_equal(
            auth["required_go"],
            {"path": GO_REL, "status_required": "GO_V3_SINGLE_TARGET_FREE_ATTEMPT"},
        ),
        "authorization required_go mismatch",
    )
    _require(_json_type_exact_equal(auth["authority_scope"], AUTHORITY_SCOPE), "authorization authority scope mismatch")
    _validate_attempt_id(auth["attempt_id"])
    _parse_utc_microseconds(auth["created_utc"], "authorization.created_utc")
    return auth, _control_identity(root, AUTH_REL)


def _validate_review(
    root: Path,
    amendment_identity: Mapping[str, Any],
    amendment_v4_identity: Mapping[str, Any],
    code_seal: Mapping[str, Any],
    code_seal_identity: Mapping[str, Any],
    auth: Mapping[str, Any],
    auth_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    review = strict_pretty_control_json_file(
        _artifact_path(root, REVIEW_REL), label=REVIEW_REL
    )
    _exact_keys(review, REVIEW_TOP_KEYS, "independent review")
    _require(type(review["schema_version"]) is int and review["schema_version"] == 3, "review schema mismatch")
    _require(review["artifact_type"] == "TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_REVIEW_V3", "review artifact mismatch")
    _require(review["status"] == "PASS_V3_EXECUTION_REVIEW_PENDING_INDEPENDENT_GO", "review status mismatch")
    _require(review["publisher"] == "INDEPENDENT_REVIEWER_APPEND_ONLY_ONLY", "review publisher mismatch")
    _require(review["verdict"] == "PASS", "review verdict mismatch")
    _require(review["attempt_id"] == auth["attempt_id"], "review attempt mismatch")
    _require(_json_type_exact_equal(review["amendment"], amendment_identity), "review amendment mismatch")
    _require(
        _json_type_exact_equal(review["amendment_v4"], amendment_v4_identity),
        "review V4 amendment mismatch",
    )
    _require(_json_type_exact_equal(review["code_seal"], code_seal_identity), "review code seal mismatch")
    _require(_json_type_exact_equal(review["authorization"], auth_identity), "review authorization mismatch")
    _require(
        _json_type_exact_equal(review["code_identities"], code_seal["code_identities"]),
        "review code identities mismatch",
    )
    _require(_json_type_exact_equal(review["output_namespace"], OUTPUT_NAMESPACE), "review output namespace mismatch")
    _require(_json_type_exact_equal(review["independent_checks"], REVIEW_CHECKS), "review independent checks mismatch")
    _require(_json_type_exact_equal(review["authority_scope"], DOCUMENTARY_SCOPE), "review authority scope mismatch")
    _parse_utc_microseconds(review["created_utc"], "review.created_utc")
    return review, _control_identity(root, REVIEW_REL)


def _validate_go(
    root: Path,
    amendment: Mapping[str, Any],
    amendment_identity: Mapping[str, Any],
    amendment_v4_identity: Mapping[str, Any],
    code_seal: Mapping[str, Any],
    code_seal_identity: Mapping[str, Any],
    auth: Mapping[str, Any],
    auth_identity: Mapping[str, Any],
    review: Mapping[str, Any],
    review_identity: Mapping[str, Any],
    go_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    go_path = _reject_linklike_components(go_path, label="independent GO path")
    _require(_same_lexical_path(go_path, _artifact_path(root, GO_REL)), "independent GO path mismatch")
    go = strict_pretty_control_json_file(go_path, label=GO_REL)
    _exact_keys(go, GO_TOP_KEYS, "independent GO")
    _require(type(go["schema_version"]) is int and go["schema_version"] == 3, "GO schema mismatch")
    _require(go["artifact_type"] == "TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_GO_V3", "GO artifact mismatch")
    _require(go["status"] == "GO_V3_SINGLE_TARGET_FREE_ATTEMPT", "GO status mismatch")
    _require(go["publisher"] == "INDEPENDENT_GO_REVIEWER_APPEND_ONLY_ONLY", "GO publisher mismatch")
    _require(go["attempt_id"] == auth["attempt_id"] == review["attempt_id"], "GO attempt mismatch")
    _require(_json_type_exact_equal(go["amendment"], amendment_identity), "GO amendment mismatch")
    _require(_json_type_exact_equal(go["amendment_v4"], amendment_v4_identity), "GO V4 amendment mismatch")
    _require(_json_type_exact_equal(go["code_seal"], code_seal_identity), "GO code-seal mismatch")
    _require(_json_type_exact_equal(go["authorization"], auth_identity), "GO authorization mismatch")
    _require(_json_type_exact_equal(go["independent_review"], review_identity), "GO review mismatch")
    _require(
        _json_type_exact_equal(go["code_identities"], code_seal["code_identities"]),
        "GO code identities mismatch",
    )
    _require(_json_type_exact_equal(go["output_namespace"], OUTPUT_NAMESPACE), "GO output namespace mismatch")
    _require(
        _json_type_exact_equal(go["required_commands"], _expected_commands(root, amendment)),
        "GO commands mismatch",
    )
    _require(_json_type_exact_equal(go["authority_scope"], GO_SCOPE), "GO authority scope mismatch")
    _require(_json_type_exact_equal(go["single_attempt"], SINGLE_ATTEMPT), "GO single attempt mismatch")
    _parse_utc_microseconds(go["created_utc"], "go.created_utc")
    return go, _control_identity(root, GO_REL)


def validate_predata_authority(
    root: Path,
    authorization_path: Path,
    independent_go_path: Path,
    *,
    repo: Path = REPO_ROOT,
    validate_bound_inputs: bool = True,
    validate_runtime: bool = True,
) -> PredataContext:
    """Validate every identity/control prerequisite before any Arrow import."""
    root = _reject_linklike_components(root, label="artifact root")
    repo = _reject_linklike_components(repo, label="repository root")
    amendment, amendment_identity = _validate_amendment(root)
    amendment_v4, amendment_v4_identity = _validate_amendment_v4(
        root,
        amendment,
        amendment_identity,
    )
    if validate_bound_inputs:
        _validate_bound_inputs(root, amendment)
    if validate_runtime:
        _validate_runtime_identity(amendment)
    code_seal, code_seal_identity = _validate_code_seal(
        root,
        repo,
        amendment,
        amendment_identity,
        amendment_v4_identity,
    )
    auth, auth_identity = _validate_authorization(
        root,
        amendment,
        amendment_identity,
        amendment_v4_identity,
        code_seal,
        code_seal_identity,
        authorization_path,
    )
    review, review_identity = _validate_review(
        root,
        amendment_identity,
        amendment_v4_identity,
        code_seal,
        code_seal_identity,
        auth,
        auth_identity,
    )
    go, go_identity = _validate_go(
        root,
        amendment,
        amendment_identity,
        amendment_v4_identity,
        code_seal,
        code_seal_identity,
        auth,
        auth_identity,
        review,
        review_identity,
        independent_go_path,
    )
    chronology = [
        _parse_utc_microseconds(amendment["created_utc"], "amendment.created_utc"),
        _parse_utc_microseconds(amendment_v4["created_utc"], "amendment_v4.created_utc"),
        _parse_utc_microseconds(code_seal["created_utc"], "code_seal.created_utc"),
        _parse_utc_microseconds(auth["created_utc"], "authorization.created_utc"),
        _parse_utc_microseconds(review["created_utc"], "review.created_utc"),
        _parse_utc_microseconds(go["created_utc"], "go.created_utc"),
    ]
    _require(all(left < right for left, right in zip(chronology, chronology[1:])), "control chronology is not strict")
    return PredataContext(
        root=root,
        repo=repo,
        amendment=amendment,
        amendment_identity=amendment_identity,
        amendment_v4=amendment_v4,
        amendment_v4_identity=amendment_v4_identity,
        code_seal=code_seal,
        code_seal_identity=code_seal_identity,
        authorization=auth,
        authorization_identity=auth_identity,
        review=review,
        review_identity=review_identity,
        independent_go=go,
        independent_go_identity=go_identity,
        attempt_id=auth["attempt_id"],
    )


def _validate_namespace_structure(
    context: PredataContext,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[Path, dict[str, Any]]]]:
    """Validate the post-runner/pre-sealer namespace without logical decoding."""
    root = context.root
    output_root = _artifact_path(root, OUTPUT_ROOT_REL)
    transaction = _artifact_path(root, TRANSACTION_REL)
    _require(output_root.is_dir() and not output_root.is_symlink(), "V3 output root missing or unsafe")
    _require(transaction.is_dir() and not transaction.is_symlink(), "V3 transaction root missing or unsafe")

    root_entries = {entry.name: entry for entry in output_root.iterdir()}
    _require(
        frozenset(root_entries) == frozenset((*ALIAS_SPECS, ".transaction_v3")),
        "post-runner output root exact3 entries mismatch",
    )
    transaction_entries = {entry.name: entry for entry in transaction.iterdir()}
    _require(
        frozenset(transaction_entries) == frozenset(RUNNER_OUTPUTS),
        "runner transaction exact7 entries mismatch",
    )
    for name, path in transaction_entries.items():
        _reject_linklike_components(path, label=f"staged output {name}")
        _require(path.is_file() and not path.is_symlink(), f"staged output unsafe: {name}")
    for forbidden in (*RUNNER_OUTPUTS, *SEALER_OUTPUTS):
        final_candidate = _reject_linklike_components(
            output_root / forbidden, label=f"final output {forbidden}"
        )
        _require(not final_candidate.exists(), f"premature final publication: {forbidden}")
    field_stage = _reject_linklike_components(
        output_root / ".alias_stage_FIELD_CENSUS_LOCK_v3",
        label="field-census alias stage",
    )
    provenance_stage = _reject_linklike_components(
        output_root / ".alias_stage_PROVENANCE_LEDGER_v3",
        label="provenance alias stage",
    )
    _require(not field_stage.exists(), "field-census alias stage remains")
    _require(not provenance_stage.exists(), "provenance alias stage remains")

    aliases: dict[str, dict[str, Any]] = {}
    amendment_aliases = context.amendment["output_contract"]["upstream_alias_closure"]["aliases"]
    for name, source_rel in ALIAS_SPECS.items():
        final = _reject_linklike_components(output_root / name, label=f"alias {name}")
        source = _artifact_path(root, source_rel)
        _require(final.is_file() and not final.is_symlink(), f"alias missing or unsafe: {name}")
        final_identity = file_identity(final, f"{OUTPUT_ROOT_REL}/{name}")
        frozen = amendment_aliases[name]
        _require(final_identity["size_bytes"] == frozen["size_bytes"], f"alias size mismatch: {name}")
        _require(final_identity["sha256"] == frozen["sha256"], f"alias sha mismatch: {name}")
        source_size, source_sha = sha256_file(source)
        _require(
            (source_size, source_sha) == (final_identity["size_bytes"], final_identity["sha256"]),
            f"alias bytes differ from source: {name}",
        )
        source_stat = source.stat()
        final_stat = final.stat()
        if source_stat.st_ino and final_stat.st_ino:
            _require(source_stat.st_ino != final_stat.st_ino, f"alias shares source inode: {name}")
        if source_stat.st_nlink:
            _require(source_stat.st_nlink == 1, f"source link count not one: {name}")
        if final_stat.st_nlink:
            _require(final_stat.st_nlink == 1, f"alias link count not one: {name}")
        aliases[name] = final_identity

    field_alias = output_root / "FIELD_CENSUS_LOCK.json"
    field_payload = strict_json_file(field_alias, label="FIELD_CENSUS_LOCK.json alias")
    _require(
        pretty_json_bytes(field_payload, ensure_ascii=False) == field_alias.read_bytes()
        or pretty_json_bytes(field_payload, ensure_ascii=True) == field_alias.read_bytes(),
        "FIELD_CENSUS_LOCK alias JSON is not deterministic pretty JSON",
    )
    _exact_keys(field_payload, FIELD_CENSUS_KEYS, "FIELD_CENSUS_LOCK alias")
    _require(field_payload["schema_version"] == 1, "FIELD_CENSUS_LOCK schema_version mismatch")
    _require(field_payload["artifact_type"] == "NOAA_GFS_MULTISEASON_CENSUS_MANIFEST", "FIELD_CENSUS_LOCK artifact_type mismatch")
    _parse_utc_microseconds(field_payload["created_utc"], "FIELD_CENSUS_LOCK.created_utc")
    _require(
        field_payload["labels_read"] is False
        and field_payload["models_fit"] == 0
        and field_payload["raw_downloaded_bytes"] == 0
        and field_payload["raw_network_requests"] == 0
        and field_payload["submission_csv_created"] is False,
        "FIELD_CENSUS_LOCK forbidden-access facts mismatch",
    )
    identity_fields = (
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
    for key in identity_fields:
        _validate_identity_record(field_payload[key], f"FIELD_CENSUS_LOCK.{key}")
    progress = field_payload["progress_checkpoints"]
    _require(isinstance(progress, list) and len(progress) == 13, "FIELD_CENSUS_LOCK progress checkpoint count mismatch")
    for ordinal, record in enumerate(progress):
        _validate_identity_record(record, f"FIELD_CENSUS_LOCK.progress_checkpoints[{ordinal}]")
    _require(progress[-2]["path"] == "census/progress/idx_files_001152.json", "FIELD_CENSUS_LOCK final object checkpoint mismatch")
    _require(progress[-1]["path"] == "census/progress/list_runs_000048.json", "FIELD_CENSUS_LOCK final run checkpoint mismatch")
    _require(field_payload["raw_download_plan"] == context.amendment["bound_inputs"]["census_raw_download_plan"], "FIELD_CENSUS_LOCK raw download plan crosslink mismatch")

    staged: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name in RUNNER_OUTPUTS:
        path = transaction / name
        staged[name] = (path, file_identity(path, f"{TRANSACTION_REL}/{name}"))
    return aliases, staged


def _import_arrow_after_predata() -> tuple[Any, Any, Any]:
    """Single import handoff used only after validate_predata_authority returns."""
    try:
        import numpy as np  # type: ignore
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific detail
        raise AuditError("required NumPy/PyArrow runtime unavailable") from exc
    return np, pa, pq


def _validate_pyarrow_write_table_signature(write_table: Any) -> None:
    parameters = inspect.signature(write_table).parameters
    expected_names = {
        "table",
        "where",
        *PARQUET_WRITE_KWARGS_EXACT,
        *PYARROW_WRITE_TABLE_UNLISTED_DEFAULTS_EXACT,
        "kwargs",
    }
    _require(set(parameters) == expected_names, "V4 PyArrow write_table signature keys drift")
    empty = inspect.Parameter.empty
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    _require(
        parameters["table"].kind is positional
        and parameters["where"].kind is positional
        and parameters["table"].default is empty
        and parameters["where"].default is empty,
        "V4 PyArrow write_table operand boundary drift",
    )
    _require(
        parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
        and parameters["kwargs"].default is empty,
        "V4 PyArrow write_table variadic keyword boundary drift",
    )
    _require(
        all(
            parameters[name].kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            for name in PARQUET_WRITE_KWARGS_EXACT
        ),
        "V4 PyArrow explicit exact12 parameter kind drift",
    )
    _require(
        {
            name: parameters[name].default
            for name in PYARROW_WRITE_TABLE_UNLISTED_DEFAULTS_EXACT
        }
        == PYARROW_WRITE_TABLE_UNLISTED_DEFAULTS_EXACT,
        "V4 PyArrow unlisted write_table defaults drift",
    )


def _validate_imported_runtime(context: PredataContext, pa: Any, pq: Any) -> None:
    try:
        import sklearn  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise AuditError("scikit-learn import failed") from exc
    expected = context.amendment["runtime_identity"]["sklearn_config"]
    observed = sklearn.get_config()
    _require(observed == expected, "scikit-learn config drift")

    _require(importlib.metadata.version("pyarrow") == "22.0.0", "V4 PyArrow runtime drift")
    _validate_pyarrow_write_table_signature(pq.write_table)
    _require(
        context.amendment_v4["corrected_parquet_serialization"]["explicit_write_table_kwargs_exact"]
        == PARQUET_WRITE_KWARGS_EXACT,
        "V4 runtime exact12 kwargs crosslink mismatch",
    )

    schema = pa.schema(
        (
            pa.field("valid_time_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("target_operating_day_kst", pa.date32(), nullable=False),
            pa.field("metric", pa.float64(), nullable=False),
        )
    )
    table = pa.Table.from_arrays(
        (
            pa.array([1_704_067_200_000_000_123], type=schema.field("valid_time_utc").type),
            pa.array([19_723], type=pa.date32()),
            pa.array([1.25], type=pa.float64()),
        ),
        schema=schema,
    )

    def write_synthetic(kwargs: Mapping[str, Any]) -> bytes:
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, **dict(kwargs))
        return sink.getvalue().to_pybytes()

    first = write_synthetic(PARQUET_WRITE_KWARGS_EXACT)
    second = write_synthetic(PARQUET_WRITE_KWARGS_EXACT)
    _require(first == second, "V4 PyArrow repeated synthetic bytes differ")
    roundtrip = pq.read_table(pa.BufferReader(first), use_threads=False)
    _require(roundtrip.schema == schema and roundtrip.equals(table), "V4 PyArrow native-ns roundtrip mismatch")
    ns_kwargs = dict(PARQUET_WRITE_KWARGS_EXACT)
    ns_kwargs["coerce_timestamps"] = "ns"
    try:
        write_synthetic(ns_kwargs)
    except ValueError as exc:
        _require(
            "Invalid value for coerce_timestamps: ns" in str(exc),
            "V4 PyArrow string-ns failure changed",
        )
    else:
        _fail("V4 PyArrow unexpectedly accepted coerce_timestamps='ns'")
    us_kwargs = dict(PARQUET_WRITE_KWARGS_EXACT)
    us_kwargs["coerce_timestamps"] = "us"
    try:
        write_synthetic(us_kwargs)
    except Exception as exc:
        _require(
            isinstance(exc, pa.ArrowInvalid),
            "V4 PyArrow forced-us failure type changed",
        )
    else:
        _fail("V4 PyArrow unexpectedly truncated native-ns residue")


def _validate_v4_affected_schema(
    pa: Any,
    schema: Any,
    basename: str,
    amendment_v4: Mapping[str, Any],
) -> None:
    contracts = amendment_v4["affected_output_schemas_exact"]
    _require(basename in contracts, f"V4 affected schema missing: {basename}")
    contract = contracts[basename]
    _require(
        contract
        == {
            "valid_time_utc": {"arrow_type": "timestamp[ns, tz=UTC]", "nullable": False},
            "target_operating_day_kst": {"arrow_type": "date32[day]", "nullable": False},
        },
        f"V4 affected schema contract drift: {basename}",
    )
    valid_time = schema.field("valid_time_utc")
    operating_day = schema.field("target_operating_day_kst")
    _require(
        pa.types.is_timestamp(valid_time.type)
        and valid_time.type.unit == "ns"
        and valid_time.type.tz == "UTC"
        and not valid_time.nullable,
        f"V4 timestamp[ns,UTC] schema mismatch: {basename}",
    )
    _require(
        pa.types.is_date32(operating_day.type) and not operating_day.nullable,
        f"V4 date32 schema mismatch: {basename}",
    )


def _parquet_metadata(pq: Any, path: Path, label: str) -> tuple[int, list[str], Any]:
    path = _reject_linklike_components(path, label=label)
    try:
        parquet = pq.ParquetFile(path)
        return parquet.metadata.num_rows, parquet.schema_arrow.names, parquet.schema_arrow
    except Exception as exc:
        raise AuditError(f"{label}: invalid parquet metadata") from exc


def _validate_source_parquet_metadata(
    context: PredataContext, pq: Any, *, validate_provenance_alias: bool = True
) -> None:
    """Footer-only checks; never calls read/read_table on an upstream parquet."""
    amendment = context.amendment
    bound = amendment["bound_inputs"]
    joins = amendment["input_and_join_contract"]
    checks = (
        (
            "decoded_site_matrix",
            joins["external_site_rows"],
            joins["external_site_columns_exact"],
        ),
        (
            "decoded_group_matrix",
            joins["external_group_rows"],
            joins["external_group_columns_exact"],
        ),
        (
            "bounded_site_predictors",
            joins["predictor_site_rows"],
            joins["bounded_site_metadata_columns_exact"] + joins["predictor_columns_exact"],
        ),
        (
            "bounded_group_predictors",
            joins["predictor_group_rows"],
            joins["bounded_group_metadata_columns_exact"] + joins["predictor_columns_exact"],
        ),
    )
    for role, expected_rows, expected_columns in checks:
        path = _resolve_bound_input(context.root, bound[role]["path"])
        rows, columns, _ = _parquet_metadata(pq, path, f"bound_inputs.{role}")
        _require(rows == expected_rows, f"bound_inputs.{role}: row count mismatch")
        _require(columns == expected_columns, f"bound_inputs.{role}: schema columns mismatch")

    if validate_provenance_alias:
        provenance = _artifact_path(context.root, OUTPUT_ROOT_REL) / "PROVENANCE_LEDGER.parquet"
        rows, columns, _ = _parquet_metadata(pq, provenance, "PROVENANCE_LEDGER alias")
        frozen = amendment["output_contract"]["upstream_alias_closure"]["aliases"]["PROVENANCE_LEDGER.parquet"]
        _require(rows == frozen["rows"] == 10_368, "PROVENANCE_LEDGER alias row count mismatch")
        required = frozen["required_preregister_provenance_fields_exact"]
        _require(all(column in columns for column in required), "PROVENANCE_LEDGER required schema missing")


def _arrow_type_matches(pa: Any, field: Any, expected: str) -> bool:
    if expected == "float64":
        return pa.types.is_float64(field.type) and not field.nullable
    if expected == "float64_nullable":
        return pa.types.is_float64(field.type) and field.nullable
    if expected == "int16":
        return pa.types.is_int16(field.type) and not field.nullable
    if expected == "bool":
        return pa.types.is_boolean(field.type) and not field.nullable
    return False


def _logical_parquet_sha256(pa: Any, table: Any) -> str:
    combined = table.combine_chunks().replace_schema_metadata(None)
    sink = pa.BufferOutputStream()
    options = pa.ipc.IpcWriteOptions(metadata_version=pa.ipc.MetadataVersion.V5)
    with pa.ipc.new_stream(sink, combined.schema, options=options) as writer:
        writer.write_table(combined)
    return sha256_bytes(sink.getvalue().to_pybytes())


def _read_staged_parquet(
    pa: Any,
    pq: Any,
    path: Path,
    schema_contract: Mapping[str, Any],
    *,
    expected_rows: int,
    label: str,
) -> tuple[Any, str]:
    rows, columns, schema = _parquet_metadata(pq, path, label)
    _require(rows == expected_rows, f"{label}: row count mismatch")
    _require(columns == schema_contract["columns_exact"], f"{label}: exact columns mismatch")
    _require(schema.metadata in (None, {}), f"{label}: schema metadata forbidden")
    numeric = schema_contract.get("numeric_types", {})
    for name, expected in numeric.items():
        _require(_arrow_type_matches(pa, schema.field(name), expected), f"{label}: type/nullability mismatch for {name}")
    try:
        table = pq.read_table(path, use_threads=False)
    except Exception as exc:
        raise AuditError(f"{label}: staged parquet logical read failed") from exc
    _require(table.num_rows == expected_rows, f"{label}: logical row count mismatch")
    _require(table.schema.equals(schema, check_metadata=True), f"{label}: footer/table schema mismatch")
    return table.combine_chunks(), _logical_parquet_sha256(pa, table)


def _read_strict_csv(
    path: Path,
    expected_columns: Sequence[str],
    expected_rows: int,
    *,
    label: str,
) -> tuple[list[dict[str, str]], bytes]:
    path = _reject_linklike_components(path, label=label)
    raw = path.read_bytes()
    _require(not raw.startswith(b"\xef\xbb\xbf"), f"{label}: BOM forbidden")
    _require(b"\r" not in raw, f"{label}: CR forbidden")
    _require(raw.endswith(b"\n"), f"{label}: terminal LF required")
    _require(b"\x00" not in raw, f"{label}: NUL forbidden")
    try:
        text = raw.decode("utf-8", errors="strict")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        _require(reader.fieldnames == list(expected_columns), f"{label}: exact header mismatch")
        _require(len(set(reader.fieldnames or ())) == len(expected_columns), f"{label}: duplicate header")
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise AuditError(f"{label}: invalid CSV") from exc
    _require(len(rows) == expected_rows, f"{label}: row count mismatch")
    forbidden = re.compile(r"^(?:[+-]?(?:nan|inf(?:inity)?))$", re.IGNORECASE)
    for ordinal, row in enumerate(rows):
        _require(None not in row, f"{label}: extra field at row {ordinal}")
        _require(frozenset(row) == frozenset(expected_columns), f"{label}: row field mismatch")
        for value in row.values():
            _require(not forbidden.fullmatch(value.strip()), f"{label}: non-finite literal at row {ordinal}")
    return rows, raw


def _read_strict_output_json(
    path: Path,
    expected_top_keys: Sequence[str],
    *,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = strict_json_bytes(raw, label=label)
    _exact_keys(value, expected_top_keys, label)
    _require(pretty_json_bytes(value, ensure_ascii=True) == raw, f"{label}: deterministic JSON serialization mismatch")
    return value, raw


def _finite_number(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label}: number required")
    result = float(value)
    _require(math.isfinite(result), f"{label}: finite number required")
    return result


def _csv_bool(value: str, label: str, *, nullable: bool = False) -> bool | None:
    if nullable and value == "":
        return None
    _require(value in ("TRUE", "FALSE"), f"{label}: uppercase boolean required")
    return value == "TRUE"


def _csv_boundary_flags(value: str, label: str) -> list[str]:
    """Parse the V4 active-name grammar; an all-FALSE vector is never serialized."""

    _require(isinstance(value, str), f"{label}: string required")
    if value == "":
        return []
    return _ordered_subset(value.split(";"), CUTOFF_NAMES, label=label, allow_empty=False)


def _csv_int(value: str, label: str) -> int:
    _require(re.fullmatch(r"0|[1-9][0-9]*", value) is not None, f"{label}: canonical nonnegative integer required")
    return int(value)


def _float_bits_equal(left: float, right: float) -> bool:
    import struct

    return struct.pack(">d", float(left)) == struct.pack(">d", float(right))


def _csv_float_pair(
    row: Mapping[str, str],
    name: str,
    *,
    label: str,
    nullable: bool = True,
) -> float | None:
    decimal = row[name]
    hex_value = row[name + "_float_hex"]
    if decimal == "" or hex_value == "":
        _require(nullable and decimal == hex_value == "", f"{label}.{name}: paired null required")
        return None
    try:
        number = float(decimal)
        from_hex = float.fromhex(hex_value)
    except ValueError as exc:
        raise AuditError(f"{label}.{name}: invalid float witness") from exc
    _require(math.isfinite(number), f"{label}.{name}: non-finite")
    if number == 0.0:
        _require(decimal == "0", f"{label}.{name}: canonical positive zero decimal required")
        _require(hex_value == "0x0.0p+0", f"{label}.{name}: canonical positive zero hex required")
    else:
        _require(hex_value == number.hex(), f"{label}.{name}: noncanonical float hex")
    _require(_float_bits_equal(number, from_hex), f"{label}.{name}: decimal/hex mismatch")
    return number


def _json_float_pair(
    record: Mapping[str, Any],
    name: str,
    *,
    label: str,
    nullable: bool = True,
) -> float | None:
    value = record[name]
    hex_value = record[name + "_float_hex"]
    if value is None or hex_value is None:
        _require(nullable and value is None and hex_value is None, f"{label}.{name}: paired null required")
        return None
    number = _finite_number(value, f"{label}.{name}")
    _require(isinstance(hex_value, str), f"{label}.{name}_float_hex: string required")
    if number == 0.0:
        _require(number == 0.0 and hex_value == "0x0.0p+0", f"{label}.{name}: canonical zero required")
    else:
        _require(hex_value == number.hex(), f"{label}.{name}: noncanonical float hex")
    _require(_float_bits_equal(number, float.fromhex(hex_value)), f"{label}.{name}: decimal/hex mismatch")
    return number


def _json_float_list_pair(record: Mapping[str, Any], name: str, *, label: str) -> list[float]:
    values = record[name]
    hex_values = record[name + "_float_hex"]
    _require(isinstance(values, list) and isinstance(hex_values, list), f"{label}.{name}: paired lists required")
    _require(len(values) == len(hex_values), f"{label}.{name}: paired list length mismatch")
    parsed: list[float] = []
    for ordinal, (value, hex_value) in enumerate(zip(values, hex_values, strict=True)):
        number = _finite_number(value, f"{label}.{name}[{ordinal}]")
        _require(isinstance(hex_value, str) and hex_value == number.hex(), f"{label}.{name}[{ordinal}]: noncanonical float hex")
        _require(_float_bits_equal(number, float.fromhex(hex_value)), f"{label}.{name}[{ordinal}]: decimal/hex mismatch")
        parsed.append(number)
    return parsed


FOLD_NAMES = ("2022_H1", "2022_H2", "2023_H1", "2023_H2")


def _fold_name_for_operating_day(value: Any) -> str:
    text = str(value)
    _require(re.fullmatch(r"20(?:22|23)-[0-9]{2}-[0-9]{2}", text) is not None, "OOF operating day is noncanonical")
    try:
        operating_day = date.fromisoformat(text)
    except ValueError as exc:
        raise AuditError("OOF operating day is invalid") from exc
    if date(2022, 1, 1) <= operating_day <= date(2022, 6, 30):
        return "2022_H1"
    if date(2022, 7, 1) <= operating_day <= date(2022, 12, 31):
        return "2022_H2"
    if date(2023, 1, 1) <= operating_day <= date(2023, 6, 30):
        return "2023_H1"
    if date(2023, 7, 1) <= operating_day <= date(2023, 12, 31):
        return "2023_H2"
    _fail("OOF operating day is outside the four frozen folds")
    raise AssertionError


def _validate_oof_table(np: Any, table: Any, amendment: Mapping[str, Any], *, production_shape: bool) -> None:
    columns = {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.schema.names}
    status = columns["slot_status"]
    invalid_reason = columns["invalid_reason"]
    actual = columns["actual"].astype(np.float64, copy=False)
    prediction = columns["oof_prediction"].astype(np.float64, copy=False)
    residual = columns["residual"].astype(np.float64, copy=False)
    allowed_status = frozenset(amendment["planned_fit_budget"]["unit_slot_status_enum_exact"])
    reasons = FIT_SKIP_REASONS
    _require(set(status.tolist()).issubset(allowed_status), "OOF slot_status outside frozen enum")
    completed = status == "COMPLETED"
    skipped = status == "SKIPPED_PREMODEL_CLEAR_VETO"
    _require(bool(np.all(completed | skipped)), "OOF status partition incomplete")
    _require(bool(np.all(np.isfinite(actual[completed]))), "OOF completed actual non-finite")
    _require(bool(np.all(np.isfinite(prediction[completed]))), "OOF completed prediction non-finite")
    _require(bool(np.all(np.isfinite(residual[completed]))), "OOF completed residual non-finite")
    _require(all(reason is None for reason in invalid_reason[completed].tolist()), "completed OOF reason must be null")
    expected_residual = np.subtract(actual[completed], prediction[completed], dtype=np.float64)
    _require(bool(np.array_equal(expected_residual, residual[completed])), "OOF residual arithmetic mismatch")
    if bool(np.any(skipped)):
        _require(bool(np.all(np.isnan(prediction[skipped]))), "skipped OOF prediction must be null")
        _require(bool(np.all(np.isnan(residual[skipped]))), "skipped OOF residual must be null")
        _require(
            all(isinstance(reason, str) and reason in reasons for reason in invalid_reason[skipped].tolist()),
            "skipped OOF reason invalid",
        )
        skipped_actual = actual[skipped]
        _require(bool(np.all(np.isfinite(skipped_actual) | np.isnan(skipped_actual))), "skipped OOF actual contains infinity")
        for value, reason in zip(skipped_actual.tolist(), invalid_reason[skipped].tolist(), strict=True):
            if math.isnan(float(value)):
                _require(
                    reason in (
                        "SKIPPED_PREMODEL_CLEAR_PHYSICAL_WIND_FAMILY_VETO",
                        "SKIPPED_PREMODEL_CLEAR_PHYSICAL_PBL_FAMILY_VETO",
                    ),
                    "null skipped OOF actual lacks physical-veto reason",
                )
    if production_shape:
        responses = amendment["planned_fit_budget"]["response_order"]
        estimators = ("RIDGE_PIPELINE", "EXTRA_TREES")
        response_values = columns["response"].tolist()
        estimator_values = columns["estimator"].tolist()
        block = 19_584
        reference_metadata = tuple(
            columns[name][:block]
            for name in (
                "valid_time_utc",
                "target_operating_day_kst",
                "forecast_hour",
                "site_id",
                "group",
                "capacity_mw",
            )
        )
        _require(bool(np.all(np.isfinite(reference_metadata[-1].astype(np.float64)))), "OOF capacity non-finite")
        expected_sites = np.tile(np.arange(1, 18, dtype=np.int64), 1_152)
        _require(bool(np.array_equal(reference_metadata[3].astype(np.int64), expected_sites)), "OOF site order mismatch")
        for response_index, response in enumerate(responses):
            for estimator_index, estimator in enumerate(estimators):
                start = (response_index * 2 + estimator_index) * block
                stop = start + block
                _require(all(item == response for item in response_values[start:stop]), "OOF response order mismatch")
                _require(all(item == estimator for item in estimator_values[start:stop]), "OOF estimator order mismatch")
                for name, expected in zip(
                    ("valid_time_utc", "target_operating_day_kst", "forecast_hour", "site_id", "group", "capacity_mw"),
                    reference_metadata,
                    strict=True,
                ):
                    _require(bool(np.array_equal(columns[name][start:stop], expected)), f"OOF canonical metadata drift: {name}")
            first = response_index * 2 * block
            second = first + block
            _require(
                bool(np.array_equal(columns["actual"][first : first + block], columns["actual"][second : second + block], equal_nan=True)),
                f"OOF actual differs between estimators: {response}",
            )
        slot_ids = columns["planned_oof_slot_id"].tolist()
        _require(len(set(slot_ids)) == len(slot_ids), "OOF slot ids not unique")
        folds = columns["fold"].tolist()
        _require(set(folds) == set(FOLD_NAMES), "OOF fold set mismatch")
        for start in range(0, len(folds), block):
            block_folds = folds[start : start + block]
            _require(
                [block_folds.count(name) for name in FOLD_NAMES] == [4_896] * 4,
                "OOF per-block fold counts mismatch",
            )
        days = columns["target_operating_day_kst"].tolist()
        _require(
            all(fold == _fold_name_for_operating_day(day) for fold, day in zip(folds, days)),
            "OOF fold does not match target operating day",
        )
        for response_index in range(len(responses)):
            for estimator_index in range(len(estimators)):
                start = (response_index * 2 + estimator_index) * block
                expected_ids = (
                    f"OOF__{response_index + 1:02d}__{estimator_index + 1:01d}__{row + 1:05d}"
                    for row in range(block)
                )
                _require(
                    all(observed == expected for observed, expected in zip(slot_ids[start : start + block], expected_ids)),
                    "OOF planned slot id/order mismatch",
                )


def _oof_site_group_map(np: Any, table: Any) -> dict[str, str]:
    site_ids = table.column("site_id").to_numpy(zero_copy_only=False)
    groups = table.column("group").to_numpy(zero_copy_only=False)
    result: dict[str, str] = {}
    for site in range(1, 18):
        observed = {str(value) for value in groups[site_ids.astype(np.int64, copy=False) == site].tolist()}
        _require(len(observed) == 1, f"OOF site/group mapping mismatch: {site:02d}")
        group = next(iter(observed))
        _require(group in ("kpx_group_1", "kpx_group_2", "kpx_group_3"), f"OOF site group enum mismatch: {site:02d}")
        result[f"{site:02d}"] = group
    return result


def _derived_source_responses(diagnostic: str) -> tuple[str, ...]:
    if diagnostic.startswith("WS_"):
        level = diagnostic.removeprefix("WS_")
        return (f"UGRD_{level}mb", f"VGRD_{level}mb")
    if diagnostic == "ENDPOINT_DU_925_MINUS_1000":
        return ("UGRD_925mb", "UGRD_1000mb")
    if diagnostic == "ENDPOINT_DV_925_MINUS_1000":
        return ("VGRD_925mb", "VGRD_1000mb")
    if diagnostic in (
        "ENDPOINT_VECTOR_SHEAR_MAG",
        "ENDPOINT_DIRECTION_COS",
        "ENDPOINT_DIRECTION_SIN",
    ):
        return ("UGRD_925mb", "VGRD_925mb", "UGRD_1000mb", "VGRD_1000mb")
    match = re.fullmatch(
        r"ADJACENT_(DU|DV|SHEAR_MAG)_(925|950|975)_MINUS_(950|975|1000)",
        diagnostic,
    )
    _require(match is not None, f"unknown derived diagnostic: {diagnostic}")
    assert match is not None
    kind, upper, lower = match.groups()
    if kind == "DU":
        return (f"UGRD_{upper}mb", f"UGRD_{lower}mb")
    if kind == "DV":
        return (f"VGRD_{upper}mb", f"VGRD_{lower}mb")
    return (
        f"UGRD_{upper}mb",
        f"VGRD_{upper}mb",
        f"UGRD_{lower}mb",
        f"VGRD_{lower}mb",
    )


def _validate_derived_table(
    np: Any,
    table: Any,
    amendment: Mapping[str, Any],
    *,
    production_shape: bool,
) -> dict[str, dict[str, str | None]]:
    columns = {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.schema.names}
    valid = columns["valid"].astype(bool, copy=False)
    actual = columns["actual"].astype(np.float64, copy=False)
    prediction = columns["oof_prediction"].astype(np.float64, copy=False)
    residual = columns["residual"].astype(np.float64, copy=False)
    _require(bool(np.all(np.isfinite(columns["capacity_mw"].astype(np.float64)))), "derived capacity non-finite")
    _require(bool(np.all(np.isfinite(actual[valid]))), "derived actual non-finite")
    _require(bool(np.all(np.isfinite(prediction[valid]))), "derived prediction non-finite")
    _require(bool(np.all(np.isfinite(residual[valid]))), "derived residual non-finite")
    _require(
        bool(np.array_equal(np.subtract(actual[valid], prediction[valid], dtype=np.float64), residual[valid])),
        "derived residual arithmetic mismatch",
    )
    if bool(np.any(~valid)):
        _require(bool(np.all(np.isfinite(actual[~valid]) | np.isnan(actual[~valid]))), "invalid derived actual contains infinity")
        _require(bool(np.all(np.isnan(prediction[~valid]))), "invalid derived prediction must be null")
        _require(bool(np.all(np.isnan(residual[~valid]))), "invalid derived residual must be null")
    reasons = columns["invalid_reason"].tolist()
    allowed_reasons = frozenset(amendment["metric_primitives"]["invalid_or_skip_reason_enum_exact"])
    _require(all(reason is None for reason, is_valid in zip(reasons, valid.tolist(), strict=True) if is_valid), "valid derived reason must be null")
    _require(
        all(isinstance(reason, str) and reason in allowed_reasons for reason, is_valid in zip(reasons, valid.tolist(), strict=True) if not is_valid),
        "invalid derived reason is outside frozen enum",
    )
    bindings = columns["source_component_estimator_binding_json"].tolist()
    parsed_bindings: list[dict[str, str | None]] = []
    for ordinal, binding in enumerate(bindings):
        _require(isinstance(binding, str) and binding != "", f"derived binding missing at {ordinal}")
        parsed = strict_json_bytes(binding.encode("ascii"), label=f"derived binding {ordinal}")
        _require(
            json.dumps(parsed, ensure_ascii=True, sort_keys=True, separators=(",", ":")) == binding,
            f"derived binding not canonical at {ordinal}",
        )
        _require(all(value in (*ESTIMATOR_ORDER, None) for value in parsed.values()), f"derived binding estimator invalid at {ordinal}")
        parsed_bindings.append(parsed)
    evidence: dict[str, dict[str, str | None]] = {}
    if production_shape:
        diagnostic_order = amendment["derived_wind_contract"]["derived_order_exact"]
        endpoint_diagnostics = frozenset(amendment["derived_wind_contract"]["endpoint_veto_diagnostics_exact"])
        _require(len(diagnostic_order) == 18, "derived diagnostic count mismatch")
        slot_ids = columns["planned_derived_slot_id"].tolist()
        _require(len(set(slot_ids)) == len(slot_ids), "derived slot ids not unique")
        for diagnostic_index, diagnostic in enumerate(diagnostic_order):
            start = diagnostic_index * 23_040
            site_stop = start + 19_584
            stop = start + 23_040
            _require(all(value == diagnostic for value in columns["diagnostic"][start:stop].tolist()), f"derived diagnostic order mismatch: {diagnostic}")
            _require(all(value == diagnostic for value in columns["formula_id"][start:stop].tolist()), f"derived formula id mismatch: {diagnostic}")
            _require(all(value == "SITE" for value in columns["scope_type"][start:site_stop].tolist()), f"derived SITE scope order mismatch: {diagnostic}")
            _require(all(value == "GROUP" for value in columns["scope_type"][site_stop:stop].tolist()), f"derived GROUP scope order mismatch: {diagnostic}")
            expected_endpoint = diagnostic in endpoint_diagnostics
            _require(bool(np.all(columns["endpoint_veto_required"][start:stop] == expected_endpoint)), f"derived endpoint-veto flag mismatch: {diagnostic}")
            binding = parsed_bindings[start]
            _exact_keys(binding, _derived_source_responses(diagnostic), f"derived source binding {diagnostic}")
            _require(all(item == binding for item in parsed_bindings[start:stop]), f"derived source binding varies within diagnostic: {diagnostic}")
            evidence[diagnostic] = binding
            expected_ids = (
                [f"DWD__{diagnostic_index + 1:02d}__S__{ordinal:05d}" for ordinal in range(1, 19_585)]
                + [f"DWD__{diagnostic_index + 1:02d}__G__{ordinal:05d}" for ordinal in range(1, 3_457)]
            )
            _require(slot_ids[start:stop] == expected_ids, f"derived slot id/order mismatch: {diagnostic}")
            expected_valid = not any(value is None for value in binding.values())
            _require(bool(np.all(valid[start:stop] == expected_valid)), f"derived validity/source-binding mismatch: {diagnostic}")
    return evidence


def _validate_fit_ledger(
    payload: Mapping[str, Any], amendment: Mapping[str, Any], *, expected_rows: int
) -> None:
    slots = payload["unit_slots"]
    _require(isinstance(slots, list) and len(slots) == expected_rows, "fit ledger unit slot count mismatch")
    frozen = amendment["planned_fit_budget"]["fit_ledger_record_schemas"]
    primary_keys = frozen["primary_model_unit_record_exact_keys"]
    affine_keys = frozen["analytic_affine_unit_record_exact_keys"]
    primary_count = 72 if expected_rows == 1332 else min(expected_rows, 1)
    identifiers: set[str] = set()
    completed = 0
    skipped = 0
    completed_primary = 0
    completed_affine = 0
    skipped_primary = 0
    skipped_affine = 0
    observed_method_calls = {
        "sklearn.pipeline.Pipeline.fit": 0,
        "sklearn.preprocessing.StandardScaler.fit": 0,
        "sklearn.linear_model.Ridge.fit": 0,
        "sklearn.ensemble.ExtraTreesRegressor.fit": 0,
    }
    allowed_reasons = FIT_SKIP_REASONS
    for ordinal, slot in enumerate(slots):
        _require(isinstance(slot, dict), f"fit ledger slot {ordinal}: object required")
        expected_keys = primary_keys if ordinal < primary_count else affine_keys
        _exact_keys(slot, expected_keys, f"fit ledger slot {ordinal}")
        for key in ("decision_fit_ordinal", "training_rows", "heldout_rows"):
            _require(
                isinstance(slot[key], int) and not isinstance(slot[key], bool) and slot[key] >= 1,
                f"fit ledger slot integer invalid: {ordinal}/{key}",
            )
        identifier = slot["planned_unit_slot_id"]
        _require(isinstance(identifier, str) and identifier not in identifiers, f"fit ledger slot id invalid: {ordinal}")
        identifiers.add(identifier)
        _require(slot["unit_status"] in ("COMPLETED", "SKIPPED_PREMODEL_CLEAR_VETO"), f"fit ledger status invalid: {ordinal}")
        _require(slot["fit_completed"] is (slot["unit_status"] == "COMPLETED"), f"fit ledger completion mismatch: {ordinal}")
        if slot["unit_status"] == "COMPLETED":
            completed += 1
            completed_primary += int(ordinal < primary_count)
            completed_affine += int(ordinal >= primary_count)
            _require(slot["skip_reason"] is None, f"completed ledger slot has skip reason: {ordinal}")
        else:
            skipped += 1
            skipped_primary += int(ordinal < primary_count)
            skipped_affine += int(ordinal >= primary_count)
            _require(slot["skip_reason"] in allowed_reasons, f"skipped ledger slot reason invalid: {ordinal}")
        if ordinal < primary_count:
            for key in (
                "training_rows",
                "training_columns",
                "heldout_rows",
                "pipeline_fit_calls",
                "standard_scaler_fit_calls",
                "ridge_fit_calls",
                "extra_trees_fit_calls",
                "predict_calls",
            ):
                _require(
                    isinstance(slot[key], int) and not isinstance(slot[key], bool) and slot[key] >= 0,
                    f"fit ledger primary integer invalid: {ordinal}/{key}",
                )
            observed_method_calls["sklearn.pipeline.Pipeline.fit"] += slot["pipeline_fit_calls"]
            observed_method_calls["sklearn.preprocessing.StandardScaler.fit"] += slot["standard_scaler_fit_calls"]
            observed_method_calls["sklearn.linear_model.Ridge.fit"] += slot["ridge_fit_calls"]
            observed_method_calls["sklearn.ensemble.ExtraTreesRegressor.fit"] += slot["extra_trees_fit_calls"]
        elif slot["unit_status"] == "COMPLETED":
            _require(isinstance(slot["constant_predictor_branch"], bool), f"completed affine branch invalid: {ordinal}")
            sxx = _json_float_pair(slot, "sxx", label=f"fit ledger slot {ordinal}")
            slope = _json_float_pair(slot, "slope", label=f"fit ledger slot {ordinal}")
            _json_float_pair(slot, "intercept", label=f"fit ledger slot {ordinal}")
            _require(sxx >= 0.0, f"affine sxx is negative: {ordinal}")
            _require(slot["constant_predictor_branch"] is (sxx <= 1e-12), f"affine constant branch mismatch: {ordinal}")
            if slot["constant_predictor_branch"]:
                _require(slope == 0.0, f"constant affine slope is not zero: {ordinal}")
        else:
            for key in (
                "sxx",
                "sxx_float_hex",
                "constant_predictor_branch",
                "slope",
                "slope_float_hex",
                "intercept",
                "intercept_float_hex",
            ):
                _require(slot[key] is None, f"skipped affine evidence must be null: {ordinal}/{key}")
    planned = payload["planned_counts"]
    _require(isinstance(planned, dict), "fit ledger planned_counts object required")
    if expected_rows == 1332:
        _require(sum(value for value in planned.values() if isinstance(value, int) and not isinstance(value, bool)) >= 1332, "fit ledger planned counts incomplete")
    executed = payload["executed_counts"]
    skipped_counts = payload["skipped_counts"]
    _require(isinstance(executed, dict) and isinstance(skipped_counts, dict), "fit ledger count maps required")
    _require(any(value == completed for value in executed.values()), "fit ledger executed count does not bind slots")
    _require(any(value == skipped for value in skipped_counts.values()), "fit ledger skipped count does not bind slots")
    expected_zero = amendment["output_contract"]["output_record_schemas_exact"]["TARGET_FREE_FIT_LEDGER_V3_json"]["zero_fit_counts"]
    _require(payload["zero_fit_counts"] == expected_zero, "fit ledger zero-fit counts mismatch")
    if expected_rows == 1332:
        responses = amendment["planned_fit_budget"]["response_order"]
        predictors = amendment["input_and_join_contract"]["predictor_columns_exact"]
        _require(
            planned
            == {
                "primary_model_units": 72,
                "analytic_affine_units": 1260,
                "total_decision_units": 1332,
                "fit_ledger_rows": 1332,
            },
            "fit ledger planned_counts mismatch",
        )
        for mapping_name, mapping in (
            ("planned_counts", planned),
            ("executed_counts", executed),
            ("skipped_counts", skipped_counts),
            ("internal_method_call_counts", payload["internal_method_call_counts"]),
            ("zero_fit_counts", payload["zero_fit_counts"]),
        ):
            _require(
                all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in mapping.values()),
                f"fit ledger {mapping_name} type/range mismatch",
            )
        _require(
            executed
            == {
                "completed_primary_model_units": completed_primary,
                "completed_analytic_affine_units": completed_affine,
                "completed_total_decision_units": completed,
            },
            "fit ledger executed_counts mismatch",
        )
        _require(
            skipped_counts
            == {
                "skipped_primary_model_units": skipped_primary,
                "skipped_analytic_affine_units": skipped_affine,
                "skipped_total_decision_units": skipped,
            },
            "fit ledger skipped_counts mismatch",
        )
        internal = dict(observed_method_calls)
        internal["total_method_calls"] = sum(observed_method_calls.values())
        _require(payload["internal_method_call_counts"] == internal, "fit ledger internal method-call counts mismatch")
        for ordinal, slot in enumerate(slots):
            _require(slot["decision_fit_ordinal"] == ordinal + 1, f"fit ledger decision ordinal mismatch: {ordinal}")
            if ordinal < 72:
                response_index, remainder = divmod(ordinal, 8)
                fold_index, estimator_index = divmod(remainder, 2)
                expected = {
                    "planned_unit_slot_id": f"PMU__{response_index + 1:02d}__{fold_index + 1}__{estimator_index + 1}",
                    "unit_kind": "PRIMARY_MODEL",
                    "response": responses[response_index],
                    "fold": FOLD_NAMES[fold_index],
                    "estimator_unit": ESTIMATOR_ORDER[estimator_index],
                }
                completed_unit = slot["unit_status"] == "COMPLETED"
                expected_calls = (
                    (1, 1, 1, 0, 1)
                    if completed_unit and estimator_index == 0
                    else (0, 0, 0, 1, 1)
                    if completed_unit
                    else (0, 0, 0, 0, 0)
                )
                _require(
                    (
                        slot["training_rows"],
                        slot["training_columns"],
                        slot["heldout_rows"],
                        slot["input_dtype"],
                        slot["response_dtype"],
                        slot["random_state"],
                    )
                    == (
                        14_688,
                        35,
                        4_896,
                        "float64_C_CONTIGUOUS" if estimator_index == 0 else "float32_C_CONTIGUOUS",
                        "float64",
                        None if estimator_index == 0 else 260_810,
                    ),
                    f"fit ledger primary execution contract mismatch: {ordinal}",
                )
                _require(
                    tuple(
                        slot[key]
                        for key in (
                            "pipeline_fit_calls",
                            "standard_scaler_fit_calls",
                            "ridge_fit_calls",
                            "extra_trees_fit_calls",
                            "predict_calls",
                        )
                    )
                    == expected_calls,
                    f"fit ledger primary call counts mismatch: {ordinal}",
                )
            else:
                affine_index = ordinal - 72
                response_index, remainder = divmod(affine_index, 140)
                predictor_index, fold_index = divmod(remainder, 4)
                expected = {
                    "planned_unit_slot_id": f"AAU__{response_index + 1:02d}__{predictor_index + 1:02d}__{fold_index + 1}",
                    "analytic_affine_ordinal": affine_index + 1,
                    "unit_kind": "ANALYTIC_AFFINE",
                    "response": responses[response_index],
                    "predictor": predictors[predictor_index],
                    "fold": FOLD_NAMES[fold_index],
                }
                _require(
                    isinstance(slot["analytic_affine_ordinal"], int)
                    and not isinstance(slot["analytic_affine_ordinal"], bool),
                    f"fit ledger affine ordinal type mismatch: {ordinal}",
                )
                _require(
                    slot["training_rows"] == 14_688 and slot["heldout_rows"] == 4_896,
                    f"fit ledger affine row counts mismatch: {ordinal}",
                )
            _require(all(slot[key] == value for key, value in expected.items()), f"fit ledger frozen slot order mismatch: {ordinal}")


def _validate_fit_ledger_oof_crosslinks(
    np: Any,
    payload: Mapping[str, Any],
    oof_table: Any,
    amendment: Mapping[str, Any],
) -> None:
    """Bind all 1,332 planned fit slots to the 352,512 OOF row slots."""
    slots = payload["unit_slots"]
    columns = {name: oof_table.column(name).to_numpy(zero_copy_only=False) for name in oof_table.schema.names}
    responses = amendment["planned_fit_budget"]["response_order"]
    block_rows = 19_584
    for response_index, response in enumerate(responses):
        primary = slots[response_index * 8 : (response_index + 1) * 8]
        affine = slots[72 + response_index * 140 : 72 + (response_index + 1) * 140]
        response_statuses = {slot["unit_status"] for slot in (*primary, *affine)}
        _require(len(response_statuses) == 1, f"fit ledger response status is not coherent: {response}")
        response_status = next(iter(response_statuses))
        response_reasons = {slot["skip_reason"] for slot in (*primary, *affine)}
        _require(
            response_reasons == ({None} if response_status == "COMPLETED" else {primary[0]["skip_reason"]}),
            f"fit ledger response skip reasons differ: {response}",
        )
        for estimator_index, estimator in enumerate(ESTIMATOR_ORDER):
            fold_slots = [primary[fold_index * 2 + estimator_index] for fold_index in range(4)]
            start = (response_index * 2 + estimator_index) * block_rows
            stop = start + block_rows
            observed_status = columns["slot_status"][start:stop]
            observed_reason = columns["invalid_reason"][start:stop]
            observed_folds = columns["fold"][start:stop]
            for fold_name, slot in zip(FOLD_NAMES, fold_slots, strict=True):
                fold_mask = observed_folds == fold_name
                _require(int(np.count_nonzero(fold_mask)) == 4_896, f"OOF fold row count mismatch: {response}/{estimator}/{fold_name}")
                _require(bool(np.all(observed_status[fold_mask] == slot["unit_status"])), f"fit-ledger/OOF status mismatch: {response}/{estimator}/{fold_name}")
                expected_reason = slot["skip_reason"]
                _require(
                    all(reason == expected_reason for reason in observed_reason[fold_mask].tolist()),
                    f"fit-ledger/OOF reason mismatch: {response}/{estimator}/{fold_name}",
                )


def _derive_manifest_oof_slot_counts(
    np: Any,
    fit_ledger: Mapping[str, Any],
    oof_table: Any,
    decision_payload: Mapping[str, Any],
) -> dict[str, int]:
    """Prove the future manifest's exact4 OOF accounting from staged evidence."""

    planned = int(fit_ledger["planned_counts"]["primary_model_units"]) * 4_896
    completed = int(
        fit_ledger["executed_counts"]["completed_primary_model_units"]
    ) * 4_896
    skipped = int(
        fit_ledger["skipped_counts"]["skipped_primary_model_units"]
    ) * 4_896
    _require(planned == 352_512, "manifest OOF planned-slot derivation mismatch")
    _require(completed + skipped == planned, "manifest OOF completed/skipped derivation mismatch")

    status = oof_table.column("slot_status").to_numpy(zero_copy_only=False)
    observed_completed = int(np.count_nonzero(status == "COMPLETED"))
    observed_skipped = int(np.count_nonzero(status == "SKIPPED_PREMODEL_CLEAR_VETO"))
    _require(observed_completed == completed, "manifest OOF completed count disagrees with staged rows")
    _require(observed_skipped == skipped, "manifest OOF skipped count disagrees with staged rows")

    selectors = sum(
        record["independent_estimator"] is not None
        for record in decision_payload["raw_component_results"]
    )
    independent_selected = selectors * 19_584
    flags = oof_table.column("independent_selected").to_numpy(zero_copy_only=False)
    _require(
        int(np.count_nonzero(flags)) == independent_selected,
        "manifest OOF selected count disagrees with decision/staged rows",
    )
    _require(
        independent_selected <= completed,
        "manifest OOF selected count exceeds completed rows",
    )
    return {
        "planned_oof_slots": planned,
        "completed_oof_slots": completed,
        "skipped_oof_slots": skipped,
        "independent_selected_oof_slots": independent_selected,
    }


def _validate_selected_oof_block_completion(
    np: Any,
    oof: Mapping[str, Any],
    start: int,
    stop: int,
    *,
    response: str,
    estimator: str,
) -> None:
    """Prove that a selected response/estimator owns all four completed folds."""

    statuses = oof["slot_status"][start:stop]
    folds = oof["fold"][start:stop]
    _require(
        stop - start == 19_584 and bool(np.all(statuses == "COMPLETED")),
        f"selected OOF block is not wholly completed: {response}/{estimator}",
    )
    _require(
        [int(np.count_nonzero(folds == fold)) for fold in FOLD_NAMES]
        == [4_896] * 4,
        f"selected OOF block does not contain four complete folds: {response}/{estimator}",
    )


def _validate_distribution(
    payload: Mapping[str, Any], amendment: Mapping[str, Any], *, expected_records: int, expected_endpoint: int
) -> dict[str, dict[str, bool]]:
    contract = amendment["output_contract"]["output_record_schemas_exact"]["TARGET_FREE_DISTRIBUTION_STABILITY_V3_json"]
    records = payload["records"]
    endpoints = payload["endpoint_pooled_metric_records"]
    _require(isinstance(records, list) and len(records) == expected_records, "distribution record count mismatch")
    _require(isinstance(endpoints, list) and len(endpoints) == expected_endpoint, "endpoint pooled record count mismatch")
    parsed_records: list[dict[str, Any]] = []
    invalid_reasons = frozenset(amendment["metric_primitives"]["invalid_or_skip_reason_enum_exact"])
    verdicts = frozenset(
        (
            "PASS_STRICT_INTERIOR",
            "CLEAR_FAIL_STRICT_INTERIOR",
            "CLEAR_FAIL_UNUSABLE",
            "AMBIGUOUS_THRESHOLD_EQUALITY",
            "STRUCTURALLY_NOT_APPLICABLE",
        )
    )
    for ordinal, record in enumerate(records):
        _require(isinstance(record, dict), f"distribution record {ordinal}: object required")
        _exact_keys(record, contract["record_keys_exact"], f"distribution record {ordinal}")
        label = f"distribution record {ordinal}"
        _require(type(record["valid"]) is bool, f"{label}: valid bool required")
        _require(record["verdict"] in verdicts, f"{label}: verdict invalid")
        iqr_2022 = _json_float_pair(record, "iqr_2022", label=label)
        iqr_2023 = _json_float_pair(record, "iqr_2023", label=label)
        iqr_ratio = _json_float_pair(record, "iqr_ratio", label=label)
        psi = _json_float_pair(record, "psi", label=label)
        _require(type(record["iqr_boundary_equal_0_1_or_10"]) is bool, f"{label}: IQR boundary bool required")
        _require(type(record["psi_boundary_equal_0_5"]) is bool, f"{label}: PSI boundary bool required")
        if record["valid"]:
            _require(record["invalid_reason"] is None, f"valid distribution record has reason: {ordinal}")
        else:
            _require(record["invalid_reason"] in invalid_reasons, f"invalid distribution record reason outside frozen enum: {ordinal}")

        kind = record["record_kind"]
        if kind == "MONTHLY_IQR":
            _require(type(record["month"]) is int and 1 <= record["month"] <= 12, f"{label}: month invalid")
            _require(record["reference_year"] is None and record["comparison_year"] is None, f"{label}: monthly years must be null")
            for key in (
                "reference_internal_edges",
                "reference_internal_edges_float_hex",
                "reference_counts",
                "comparison_counts",
                "reference_probabilities",
                "reference_probabilities_float_hex",
                "comparison_probabilities",
                "comparison_probabilities_float_hex",
            ):
                _require(record[key] == [], f"{label}: monthly {key} must be empty")
            _require(record["final_bin_count"] is None and psi is None, f"{label}: monthly PSI fields must be null")
            _require(record["psi_boundary_equal_0_5"] is False, f"{label}: monthly PSI boundary must be false")
            if record["valid"]:
                _require(iqr_2022 is not None and iqr_2023 is not None and iqr_ratio is not None, f"{label}: valid IQR metrics missing")
                boundary = iqr_ratio in (0.1, 10.0)
                passes = 0.1 < iqr_ratio < 10.0
                _require(record["iqr_boundary_equal_0_1_or_10"] is boundary, f"{label}: IQR boundary mismatch")
                expected_verdict = "AMBIGUOUS_THRESHOLD_EQUALITY" if boundary else "PASS_STRICT_INTERIOR" if passes else "CLEAR_FAIL_STRICT_INTERIOR"
            else:
                _require(record["iqr_boundary_equal_0_1_or_10"] is False, f"{label}: invalid IQR boundary must be false")
                structural = iqr_2022 is None and iqr_2023 is None and iqr_ratio is None
                expected_verdict = "STRUCTURALLY_NOT_APPLICABLE" if structural else "CLEAR_FAIL_UNUSABLE"
        elif kind == "YEAR_PSI":
            _require(record["month"] is None, f"{label}: PSI month must be null")
            _require(record["reference_year"] == 2022 and record["comparison_year"] == 2023, f"{label}: PSI years mismatch")
            _require(iqr_2022 is None and iqr_2023 is None and iqr_ratio is None, f"{label}: PSI IQR fields must be null")
            _require(record["iqr_boundary_equal_0_1_or_10"] is False, f"{label}: PSI IQR boundary must be false")
            edges = _json_float_list_pair(record, "reference_internal_edges", label=label)
            p = _json_float_list_pair(record, "reference_probabilities", label=label)
            q = _json_float_list_pair(record, "comparison_probabilities", label=label)
            counts_p = record["reference_counts"]
            counts_q = record["comparison_counts"]
            _require(isinstance(counts_p, list) and isinstance(counts_q, list), f"{label}: PSI count lists required")
            _require(all(type(value) is int and value >= 0 for value in (*counts_p, *counts_q)), f"{label}: PSI counts invalid")
            if record["valid"]:
                bins = record["final_bin_count"]
                _require(type(bins) is int and bins >= 2, f"{label}: PSI final bin count invalid")
                _require(len(edges) == bins - 1 and len(p) == len(q) == len(counts_p) == len(counts_q) == bins, f"{label}: PSI vector length mismatch")
                _require(edges == sorted(set(edges)), f"{label}: PSI edges not strictly ordered")
                _require(all(value > 0 for value in (*p, *q)), f"{label}: PSI probabilities invalid")
                _require(abs(math.fsum(p) - 1.0) <= 1e-12, f"{label}: PSI reference probabilities do not sum to one")
                _require(abs(math.fsum(q) - 1.0) <= 1e-12, f"{label}: PSI comparison probabilities do not sum to one")
                recomputed = math.fsum((b - a) * math.log(b / a) for a, b in zip(p, q, strict=True))
                _require(psi is not None and _float_bits_equal(recomputed, psi), f"{label}: PSI arithmetic mismatch")
                boundary = psi == 0.5
                passes = psi < 0.5
                _require(record["psi_boundary_equal_0_5"] is boundary, f"{label}: PSI boundary mismatch")
                expected_verdict = "AMBIGUOUS_THRESHOLD_EQUALITY" if boundary else "PASS_STRICT_INTERIOR" if passes else "CLEAR_FAIL_STRICT_INTERIOR"
            else:
                _require(record["psi_boundary_equal_0_5"] is False, f"{label}: invalid PSI boundary must be false")
                structural = record["final_bin_count"] is None and not edges and not p and not q and psi is None
                expected_verdict = "STRUCTURALLY_NOT_APPLICABLE" if structural else "CLEAR_FAIL_UNUSABLE"
        else:
            _fail(f"{label}: record_kind invalid")
        _require(record["verdict"] == expected_verdict, f"{label}: verdict arithmetic mismatch")
        parsed_records.append({"record": record, "boundary": expected_verdict == "AMBIGUOUS_THRESHOLD_EQUALITY"})
    for ordinal, record in enumerate(endpoints):
        _require(isinstance(record, dict), f"endpoint pooled record {ordinal}: object required")
        _exact_keys(record, contract["endpoint_pooled_metric_record_keys_exact"], f"endpoint pooled record {ordinal}")
        for key in ("sse", "rmse", "r2", "nrmse", "actual_q05", "actual_q95", "nrmse_denominator"):
            _json_float_pair(record, key, label=f"endpoint pooled record {ordinal}")
        _require(type(record["valid"]) is bool, f"endpoint pooled record {ordinal}: valid bool required")
        _require(type(record["strict_stable_margin_pass"]) is bool, f"endpoint pooled record {ordinal}: strict pass bool required")
        _require(type(record["r2_boundary_equal_0_9925"]) is bool, f"endpoint pooled record {ordinal}: r2 boundary bool required")
        _require(type(record["nrmse_boundary_equal_0_025"]) is bool, f"endpoint pooled record {ordinal}: nrmse boundary bool required")
    evidence: dict[str, dict[str, bool]] = {}
    if expected_records == 624:
        diagnostics = amendment["planned_fit_budget"]["response_order"] + contract["endpoint_pooled_diagnostic_order"]
        scopes = ["SITE_POOLED", "kpx_group_1", "kpx_group_2", "kpx_group_3"]
        _require(payload["diagnostic_order"] == diagnostics, "distribution diagnostic order mismatch")
        _require(payload["scope_order"] == scopes, "distribution scope order mismatch")
        for diagnostic_index, diagnostic in enumerate(diagnostics):
            block = parsed_records[diagnostic_index * 52 : (diagnostic_index + 1) * 52]
            _require(len(block) == 52, f"distribution block length mismatch: {diagnostic}")
            expected_order = [(scope, "MONTHLY_IQR", month) for scope in scopes for month in range(1, 13)]
            expected_order.extend((scope, "YEAR_PSI", None) for scope in scopes)
            expected_order = [
                item
                for scope in scopes
                for item in [*((scope, "MONTHLY_IQR", month) for month in range(1, 13)), (scope, "YEAR_PSI", None)]
            ]
            observed_order = [(item["record"]["scope"], item["record"]["record_kind"], item["record"]["month"]) for item in block]
            _require(all(item["record"]["diagnostic"] == diagnostic for item in block), f"distribution diagnostic order mismatch: {diagnostic}")
            _require(observed_order == expected_order, f"distribution scope/kind/month order mismatch: {diagnostic}")
            evidence[diagnostic] = {
                "pass": all(item["record"]["verdict"] == "PASS_STRICT_INTERIOR" for item in block),
                "boundary": any(item["boundary"] for item in block),
                "available": any(item["record"]["verdict"] != "STRUCTURALLY_NOT_APPLICABLE" for item in block),
            }
    return evidence


def _ordered_subset(values: Any, frozen: Sequence[str], *, label: str, allow_empty: bool = True) -> list[str]:
    _require(isinstance(values, list), f"{label}: list required")
    _require(len(values) == len(set(values)), f"{label}: duplicate value")
    _require(values == [item for item in frozen if item in values], f"{label}: value/order mismatch")
    if not allow_empty:
        _require(bool(values), f"{label}: empty list forbidden")
    return values


def _endpoint_source_responses(diagnostic: str) -> tuple[str, ...]:
    mapping = {
        "ENDPOINT_DU_925_MINUS_1000": ("UGRD_925mb", "UGRD_1000mb"),
        "ENDPOINT_DV_925_MINUS_1000": ("VGRD_925mb", "VGRD_1000mb"),
        "ENDPOINT_VECTOR_SHEAR_MAG": (
            "UGRD_925mb",
            "VGRD_925mb",
            "UGRD_1000mb",
            "VGRD_1000mb",
        ),
    }
    _require(diagnostic in mapping, f"unknown endpoint diagnostic: {diagnostic}")
    return mapping[diagnostic]


def _threshold_scope_sort_key(
    scope: Any,
    diagnostic: str,
    cutoff_name: str,
    *,
    responses: Sequence[str],
    endpoints: Sequence[str],
    predictors: Sequence[str],
) -> tuple[int, int]:
    _require(isinstance(scope, str), "threshold scope string required")
    parts = scope.split("/")
    kind_rank: tuple[int, int] | None = None
    if len(parts) == 4 and parts[:2] == ["RAW", diagnostic] and diagnostic in responses:
        if parts[2] == "AFFINE" and parts[3] in predictors:
            _require(cutoff_name.startswith("AFFINE_"), "affine cutoff/scope mismatch")
            kind_rank = (0, predictors.index(parts[3]))
        elif parts[2] == "PARENT" and parts[3] in ESTIMATOR_ORDER:
            _require(cutoff_name.startswith("PARENT_"), "parent cutoff/scope mismatch")
            kind_rank = (1, ESTIMATOR_ORDER.index(parts[3]))
        elif parts[2] == "INDEPENDENT" and parts[3] in ESTIMATOR_ORDER:
            _require(cutoff_name.startswith("INDEPENDENT_"), "independent cutoff/scope mismatch")
            kind_rank = (2, ESTIMATOR_ORDER.index(parts[3]))
    if len(parts) == 5 and parts[0] in ("RAW", "ENDPOINT") and parts[1] == diagnostic and parts[2] == "CELL":
        _require(cutoff_name.startswith("CELL_"), "cell cutoff/scope mismatch")
        labels = {
            "YEAR": ("2022", "2023"),
            "SEASON": ("DJF", "MAM", "JJA", "SON"),
            "FORECAST_HOUR": ("EARLY", "MIDDLE", "LATE"),
            "GROUP": ("kpx_group_1", "kpx_group_2", "kpx_group_3"),
            "SITE": tuple(f"{index:02d}" for index in range(1, 18)),
        }
        order = ("YEAR", "SEASON", "FORECAST_HOUR", "GROUP", "SITE")
        if parts[3] in labels and parts[4] in labels[parts[3]]:
            kind_rank = (10 + order.index(parts[3]), labels[parts[3]].index(parts[4]))
    if len(parts) == 6 and parts[0] in ("RAW", "ENDPOINT") and parts[1] == diagnostic and parts[2] == "DISTRIBUTION":
        scopes = ("SITE_POOLED", "kpx_group_1", "kpx_group_2", "kpx_group_3")
        if parts[3] in scopes and parts[4] == "MONTH" and parts[5] in tuple(f"{month:02d}" for month in range(1, 13)):
            _require(cutoff_name.startswith("MONTHLY_IQR_RATIO_"), "IQR cutoff/scope mismatch")
            kind_rank = (20 + scopes.index(parts[3]), int(parts[5]) - 1)
        elif parts[3] in scopes and parts[4:] == ["YEAR", "2022_2023"]:
            _require(cutoff_name == "YEAR_PSI_0.5", "PSI cutoff/scope mismatch")
            kind_rank = (24, scopes.index(parts[3]))
    if parts == ["ENDPOINT", diagnostic, "POOLED"] and diagnostic in endpoints:
        _require(
            cutoff_name in ("INDEPENDENT_R2_STABLE_MARGIN_0.9925", "INDEPENDENT_NRMSE_STABLE_MARGIN_0.025"),
            "endpoint pooled cutoff/scope mismatch",
        )
        kind_rank = (30, 0)
    _require(kind_rank is not None, f"threshold scope grammar mismatch: {scope}")
    return kind_rank


def _canonical_finite_hex(value: Any, *, label: str) -> float:
    _require(isinstance(value, str), f"{label}: float hex string required")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise AuditError(f"{label}: invalid float hex") from exc
    _require(math.isfinite(parsed), f"{label}: non-finite float hex")
    _require(value == parsed.hex(), f"{label}: noncanonical float hex")
    return parsed


def _wind_pressure_level(response: str) -> int:
    match = WIND_RESPONSE_RE.fullmatch(response)
    _require(match is not None, f"invalid wind response: {response}")
    return int(match.group(1))


def _wind_passes_cover_required_axes_and_levels(wind_passes: Sequence[str]) -> bool:
    levels = {_wind_pressure_level(response) for response in wind_passes}
    return (
        len(wind_passes) >= 4
        and any(response.startswith("UGRD_") for response in wind_passes)
        and any(response.startswith("VGRD_") for response in wind_passes)
        and len(levels) >= 2
    )


def _expected_global_ambiguity_reasons(
    raw_results: Sequence[Mapping[str, Any]],
    threshold_witnesses: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    reasons.extend(
        f"AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE:{record['response']}"
        for record in raw_results
        if record["classification"] == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
    )
    reasons.extend(
        "AMBIGUOUS_THRESHOLD_EQUALITY:"
        f"{witness['cutoff_name']}:{witness['scope']}:{witness['diagnostic']}:{witness['value_float_hex']}"
        for witness in threshold_witnesses
        if witness["boundary_equal"] is True
    )
    reasons.extend(
        f"AMBIGUOUS_PARENT_ESTIMATOR_TIE:{record['response']}"
        for record in raw_results
        if record["classification"] == "AMBIGUOUS_PARENT_ESTIMATOR_TIE"
    )
    reasons.extend(
        f"AMBIGUOUS_GRAY_ZONE:{record['response']}"
        for record in raw_results
        if record["classification"] == "AMBIGUOUS_GRAY_ZONE"
    )
    return reasons


def _validate_decision(payload: Mapping[str, Any], amendment: Mapping[str, Any]) -> dict[str, Any]:
    statuses = amendment["output_contract"]["decision_status_literals_exact"]
    responses = tuple(amendment["planned_fit_budget"]["response_order"])
    endpoints = tuple(amendment["derived_wind_contract"]["endpoint_veto_diagnostics_exact"])
    predictors = tuple(amendment["input_and_join_contract"]["predictor_columns_exact"])
    _require(payload["schema_version"] == 3, "decision schema_version mismatch")
    _require(payload["artifact_type"] == "TARGET_FREE_FAMILY_DECISION_V3", "decision artifact_type mismatch")
    _require(payload["status"] in statuses.values(), "decision status outside frozen enum")
    _require(payload["raw_component_order"] == list(responses), "decision raw component order mismatch")

    raw_results = payload["raw_component_results"]
    _require(isinstance(raw_results, list) and len(raw_results) == len(responses) == 9, "raw component result count mismatch")
    raw_by_response: dict[str, Mapping[str, Any]] = {}
    for ordinal, (record, response) in enumerate(zip(raw_results, responses, strict=True)):
        _require(isinstance(record, dict), f"raw component {ordinal}: object required")
        _exact_keys(record, RAW_COMPONENT_RESULT_KEYS, f"raw component {ordinal}")
        _require(record["response"] == response, f"raw component order mismatch: {ordinal}")
        classification = record["classification"]
        _require(classification in RAW_CLASSIFICATIONS, f"raw component classification invalid: {response}")
        _require(record["counts_as_nonredundant"] is (classification == "PASS_NONREDUNDANT_MARGIN"), f"raw component count relation mismatch: {response}")
        no_selector = classification in PREMODEL_CLASSIFICATIONS or classification == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
        if no_selector:
            _require(record["independent_estimator"] is None, f"raw selector must be null: {response}")
        else:
            _require(record["independent_estimator"] in ESTIMATOR_ORDER, f"raw selector invalid: {response}")
        parents = _ordered_subset(record["parent_reference_estimators"], ESTIMATOR_ORDER, label=f"raw parent witnesses {response}", allow_empty=no_selector)
        _require(bool(parents) is (not no_selector), f"raw parent witness cardinality mismatch: {response}")
        ambiguity = classification in AMBIGUOUS_CLASSIFICATIONS
        for key in ("parent_redundancy_boolean", "cell_stability_pass", "distribution_stability_pass"):
            observed = record[key]
            masked_parent_tie = (
                key == "parent_redundancy_boolean"
                and classification == "CLEAR_REDUNDANT_AFFINE"
                and len(parents) == 2
            )
            if classification in PREMODEL_CLASSIFICATIONS or ambiguity or masked_parent_tie:
                _require(observed is None, f"raw unresolved {key} must be null: {response}")
            else:
                _require(type(observed) is bool, f"raw completed {key} must be bool: {response}")
        affine = _ordered_subset(record["affine_qualifying_predictors"], predictors, label=f"raw affine witnesses {response}")
        if no_selector:
            _require(not affine, f"raw affine witnesses must be empty: {response}")
        if classification == "CLEAR_REDUNDANT_AFFINE":
            _require(bool(affine), f"affine classification requires a qualifying predictor: {response}")
        flags = _ordered_subset(record["boundary_equality_flags"], CUTOFF_NAMES, label=f"raw boundary flags {response}")
        _require(bool(flags) is (classification == "AMBIGUOUS_THRESHOLD_EQUALITY"), f"raw threshold flag/classification mismatch: {response}")
        if ambiguity:
            _require(record["ambiguity_reason"] == classification, f"raw ambiguity reason mismatch: {response}")
        else:
            _require(record["ambiguity_reason"] is None, f"raw clear ambiguity reason must be null: {response}")
        raw_by_response[response] = record

    endpoint_results = payload["endpoint_veto_results"]
    _require(isinstance(endpoint_results, list) and len(endpoint_results) == len(endpoints) == 3, "endpoint result count mismatch")
    endpoint_by_diagnostic: dict[str, Mapping[str, Any]] = {}
    for ordinal, (record, diagnostic) in enumerate(zip(endpoint_results, endpoints, strict=True)):
        _require(isinstance(record, dict), f"endpoint {ordinal}: object required")
        _exact_keys(record, ENDPOINT_VETO_RESULT_KEYS, f"endpoint {ordinal}")
        _require(record["diagnostic"] == diagnostic, f"endpoint order mismatch: {ordinal}")
        verdict = record["verdict"]
        _require(verdict in ENDPOINT_VERDICTS, f"endpoint verdict invalid: {diagnostic}")
        sources = _endpoint_source_responses(diagnostic)
        binding = record["source_component_estimator_binding"]
        _require(isinstance(binding, dict), f"endpoint binding object required: {diagnostic}")
        _exact_keys(binding, sources, f"endpoint binding {diagnostic}")
        expected_binding = {
            response: (
                None
                if raw_by_response[response]["classification"] in PREMODEL_CLASSIFICATIONS
                or raw_by_response[response]["classification"] == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
                else raw_by_response[response]["independent_estimator"]
            )
            for response in sources
        }
        _require(binding == expected_binding, f"endpoint/raw selector binding mismatch: {diagnostic}")
        source_premodel = any(raw_by_response[response]["classification"] in PREMODEL_CLASSIFICATIONS for response in sources)
        source_tie = any(raw_by_response[response]["classification"] == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE" for response in sources)
        if source_tie:
            _require(verdict == "AMBIGUOUS_RAW_ESTIMATOR_TIE", f"endpoint raw tie verdict mismatch: {diagnostic}")
        elif source_premodel:
            _require(verdict == "CLEAR_ENDPOINT_VETO_FAILURE", f"endpoint premodel verdict mismatch: {diagnostic}")
        else:
            _require(all(value in ESTIMATOR_ORDER for value in binding.values()), f"endpoint selector missing: {diagnostic}")
            _require(verdict not in ("AMBIGUOUS_RAW_ESTIMATOR_TIE",), f"endpoint raw tie verdict without tie: {diagnostic}")
        aggregates = [record[key] for key in ("pooled_stable_margin_pass", "cell_stability_pass", "distribution_stability_pass")]
        flags = _ordered_subset(record["boundary_equality_flags"], CUTOFF_NAMES, label=f"endpoint boundary flags {diagnostic}")
        if source_premodel or source_tie:
            _require(all(value is None for value in aggregates) and not flags, f"endpoint unavailable metrics must be null: {diagnostic}")
        elif verdict == "AMBIGUOUS_THRESHOLD_EQUALITY":
            _require(bool(flags) and any(value is None for value in aggregates), f"endpoint threshold evidence mismatch: {diagnostic}")
            _require(all(value is None or type(value) is bool for value in aggregates), f"endpoint aggregate type mismatch: {diagnostic}")
        else:
            _require(all(type(value) is bool for value in aggregates) and not flags, f"endpoint completed aggregate mismatch: {diagnostic}")
        _require(record["clear_pass"] is (verdict == "PASS_ENDPOINT_NONREDUNDANT_MARGIN"), f"endpoint clear_pass mismatch: {diagnostic}")
        ambiguous_endpoint = verdict in ("AMBIGUOUS_THRESHOLD_EQUALITY", "AMBIGUOUS_RAW_ESTIMATOR_TIE")
        if ambiguous_endpoint:
            _require(record["ambiguity_reason"] == verdict, f"endpoint ambiguity reason mismatch: {diagnostic}")
        else:
            _require(record["ambiguity_reason"] is None, f"endpoint clear ambiguity reason must be null: {diagnostic}")
        endpoint_by_diagnostic[diagnostic] = record

    thresholds = payload["all_threshold_float_hex_witnesses"]
    _require(isinstance(thresholds, list), "threshold witnesses list required")
    threshold_seen: set[tuple[str, str, str]] = set()
    threshold_order: list[tuple[int, int, tuple[int, int]]] = []
    boundary_by_diagnostic: dict[str, set[str]] = {name: set() for name in (*responses, *endpoints)}
    diagnostics = (*responses, *endpoints)
    for ordinal, witness in enumerate(thresholds):
        _require(isinstance(witness, dict), f"threshold witness {ordinal}: object required")
        _exact_keys(witness, THRESHOLD_WITNESS_KEYS, f"threshold witness {ordinal}")
        cutoff_name = witness["cutoff_name"]
        diagnostic = witness["diagnostic"]
        _require(cutoff_name in CUTOFF_VALUES and diagnostic in diagnostics, f"threshold witness enum mismatch: {ordinal}")
        value = _json_float_pair(witness, "value", label=f"threshold witness {ordinal}", nullable=False)
        assert value is not None
        cutoff = CUTOFF_VALUES[cutoff_name]
        _require(witness["cutoff_float_hex"] == float(cutoff).hex(), f"threshold cutoff hex mismatch: {ordinal}")
        _require(type(witness["boundary_equal"]) is bool and witness["boundary_equal"] is (value == cutoff), f"threshold equality mismatch: {ordinal}")
        key = (cutoff_name, witness["scope"], diagnostic)
        _require(key not in threshold_seen, f"duplicate threshold witness: {ordinal}")
        threshold_seen.add(key)
        scope_key = _threshold_scope_sort_key(
            witness["scope"], diagnostic, cutoff_name, responses=responses, endpoints=endpoints, predictors=predictors
        )
        threshold_order.append((CUTOFF_NAMES.index(cutoff_name), diagnostics.index(diagnostic), scope_key))
        if witness["boundary_equal"]:
            boundary_by_diagnostic[diagnostic].add(cutoff_name)
    _require(threshold_order == sorted(threshold_order), "threshold witness frozen order mismatch")
    for diagnostic, record in {**raw_by_response, **endpoint_by_diagnostic}.items():
        expected_flags = [name for name in CUTOFF_NAMES if name in boundary_by_diagnostic[diagnostic]]
        _require(record["boundary_equality_flags"] == expected_flags, f"threshold flag/witness crosslink mismatch: {diagnostic}")

    ties = payload["all_tied_witnesses"]
    _require(isinstance(ties, list), "tied witnesses list required")
    tie_seen: set[tuple[str, str]] = set()
    tie_order: list[tuple[int, int]] = []
    tie_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for ordinal, witness in enumerate(ties):
        _require(isinstance(witness, dict), f"tie witness {ordinal}: object required")
        _exact_keys(witness, TIED_WITNESS_KEYS, f"tie witness {ordinal}")
        kind = witness["tie_kind"]
        diagnostic = witness["diagnostic"]
        _require(kind in TIE_KINDS and diagnostic in responses, f"tie witness enum mismatch: {ordinal}")
        _require(witness["estimators"] == list(ESTIMATOR_ORDER), f"tie estimator order mismatch: {ordinal}")
        _require(witness["metric_name"] == ("RMSE" if kind == "INDEPENDENT_RMSE" else "R2"), f"tie metric mismatch: {ordinal}")
        hexes = witness["metric_values_float_hex"]
        _require(isinstance(hexes, list) and len(hexes) == 2, f"tie metric cardinality mismatch: {ordinal}")
        parsed = [_canonical_finite_hex(item, label=f"tie witness {ordinal}") for item in hexes]
        _require(_float_bits_equal(parsed[0], parsed[1]), f"tied metrics are not exactly equal: {ordinal}")
        resolution = witness["resolution"]
        _require(resolution in TIE_RESOLUTIONS, f"tie resolution invalid: {ordinal}")
        _require(type(witness["global_ambiguity"]) is bool and witness["global_ambiguity"] is (resolution == "GLOBAL_AMBIGUITY"), f"tie resolution/global relation mismatch: {ordinal}")
        key = (kind, diagnostic)
        _require(key not in tie_seen, f"duplicate tie witness: {ordinal}")
        tie_seen.add(key)
        tie_by_key[key] = witness
        tie_order.append((responses.index(diagnostic), TIE_KINDS.index(kind)))
    _require(tie_order == sorted(tie_order), "tie witness frozen order mismatch")
    for response, record in raw_by_response.items():
        independent_tie = ("INDEPENDENT_RMSE", response) in tie_by_key
        _require(independent_tie is (record["classification"] == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"), f"independent tie/classification mismatch: {response}")
        parent_tie = tie_by_key.get(("PARENT_MAX_R2", response))
        _require((parent_tie is not None) is (len(record["parent_reference_estimators"]) == 2), f"parent tie/witness mismatch: {response}")
        if parent_tie is not None:
            if record["classification"] == "AMBIGUOUS_PARENT_ESTIMATOR_TIE":
                _require(parent_tie["resolution"] == "GLOBAL_AMBIGUITY", f"parent ambiguity resolution mismatch: {response}")
            elif parent_tie["resolution"] == "GLOBAL_AMBIGUITY":
                _fail(f"spurious global parent tie resolution: {response}")
            elif record["parent_redundancy_boolean"] is not None:
                expected_resolution = "SHARED_PARENT_REDUNDANCY_TRUE" if record["parent_redundancy_boolean"] else "SHARED_PARENT_REDUNDANCY_FALSE"
                _require(parent_tie["resolution"] == expected_resolution, f"parent shared resolution mismatch: {response}")
            else:
                _require(
                    record["classification"] == "CLEAR_REDUNDANT_AFFINE"
                    and bool(record["affine_qualifying_predictors"])
                    and parent_tie["resolution"] == "MASKED_BY_PRIOR_AFFINE_REDUNDANCY",
                    f"masked parent tie resolution mismatch: {response}",
                )

    expected_reasons = _expected_global_ambiguity_reasons(raw_results, thresholds)
    _require(payload["global_ambiguity_reasons"] == expected_reasons, "global ambiguity reason/order mismatch")
    ambiguity = bool(expected_reasons)
    _require(payload["global_ambiguity"] is ambiguity, "decision global ambiguity mismatch")

    family_clear = payload["family_clear_pass_results"]
    _require(isinstance(family_clear, dict), "family clear results object required")
    _exact_keys(family_clear, FIXED_FAMILY_PRIORITY, "family clear results")
    _require(all(type(value) is bool for value in family_clear.values()), "family clear result bool required")
    pbl_clear = raw_by_response["HPBL_surface"]["classification"] == "PASS_NONREDUNDANT_MARGIN"
    wind_records = [raw_by_response[name] for name in responses if name != "HPBL_surface"]
    wind_passes = [record["response"] for record in wind_records if record["classification"] == "PASS_NONREDUNDANT_MARGIN"]
    wind_clear = (
        not any(record["classification"] in AMBIGUOUS_CLASSIFICATIONS for record in wind_records)
        and not any(record["classification"] == "CLEAR_PHYSICAL_OR_COVERAGE_FAILURE" for record in wind_records)
        and _wind_passes_cover_required_axes_and_levels(wind_passes)
        and all(record["clear_pass"] is True for record in endpoint_results)
    )
    _require(family_clear == {"LOW_LEVEL_ISOBARIC_WIND_PROFILE": wind_clear, "PBL_HEIGHT": pbl_clear}, "family clear recomputation mismatch")
    _require(payload["fixed_priority"] == list(FIXED_FAMILY_PRIORITY), "decision fixed priority mismatch")

    selected = payload["selected_family"]
    selected_raw = payload["selected_raw_columns"]
    selected_derived = payload["selected_derived_columns"]
    wind_raw = [name for name in responses if name != "HPBL_surface"]
    derived_order = amendment["derived_wind_contract"]["derived_order_exact"]
    if ambiguity:
        _require(payload["status"] == statuses["global_ambiguity"] and selected is None and selected_raw == [] and selected_derived == [], "ambiguous decision terminal closure invalid")
    elif wind_clear:
        _require(payload["status"] == statuses["positive"] and selected == "LOW_LEVEL_ISOBARIC_WIND_PROFILE" and selected_raw == wind_raw and selected_derived == derived_order, "wind fixed-priority terminal closure invalid")
    elif pbl_clear:
        _require(payload["status"] == statuses["positive"] and selected == "PBL_HEIGHT" and selected_raw == ["HPBL_surface"] and selected_derived == [], "PBL terminal closure invalid")
    else:
        _require(payload["status"] == statuses["clear_no_family"] and selected is None and selected_raw == [] and selected_derived == [], "clear no-family terminal closure invalid")

    expected_attestation = {
        "labels_read": False,
        "arrays_2024_2025_read": False,
        "network_requests": 0,
        "generation_model_fit_units": 0,
        "submission_csv_files_written": 0,
    }
    _require(payload["no_label_network_future_year_model_submission_attestation"] == expected_attestation, "decision forbidden-access attestation invalid")
    return {"status": payload["status"], "selected_family": selected, "global_ambiguity": ambiguity}


def _load_bound_parent_plan(
    root: Path, amendment: Mapping[str, Any]
) -> dict[str, Any]:
    record = amendment["bound_inputs"]["parent_plan"]
    identity = {key: record[key] for key in FILE_IDENTITY_KEYS}
    path = _resolve_bound_input(root, record["path"])
    verify_file_identity(
        identity,
        path,
        label="bound_inputs.parent_plan",
        expected_path=record["path"],
    )
    parent = strict_json_file(path, label="bound_inputs.parent_plan")
    families = parent.get("candidate_families")
    _require(isinstance(families, list) and len(families) == 2, "parent plan candidate family closure mismatch")
    expected_by_family = {
        "PBL_HEIGHT": [GRIB_SELECTOR_BY_RAW_COLUMN["HPBL_surface"]],
        "LOW_LEVEL_ISOBARIC_WIND_PROFILE": [
            GRIB_SELECTOR_BY_RAW_COLUMN[name]
            for name in amendment["planned_fit_budget"]["response_order"]
            if name != "HPBL_surface"
        ],
    }
    observed: dict[str, Any] = {}
    for item in families:
        _require(isinstance(item, dict), "parent plan candidate family record invalid")
        family = item.get("family")
        _require(family in expected_by_family and family not in observed, "parent plan candidate family order/set mismatch")
        observed[family] = item.get("selectors")
    _require(observed == expected_by_family, "parent plan GRIB selector closure mismatch")
    return parent


def _lock_static_sections(
    root: Path, amendment: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_contract = amendment["input_and_join_contract"]
    site_group_scope = {key: input_contract[key] for key in SITE_GROUP_SCOPE_KEYS}
    parent = _load_bound_parent_plan(root, amendment)
    spatial_transforms = {
        "spatial_method": parent["decode_contract"]["spatial_method"],
        "group_construction": amendment["derived_wind_contract"]["group_construction"],
    }
    _exact_keys(site_group_scope, SITE_GROUP_SCOPE_KEYS, "family lock site_group_scope")
    _exact_keys(spatial_transforms, SPATIAL_TRANSFORM_KEYS, "family lock spatial_transforms")
    return site_group_scope, spatial_transforms


def _selected_units(
    raw_columns: Sequence[str], derived_columns: Sequence[str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in raw_columns:
        _require(name in RAW_COLUMN_UNITS, f"family lock raw unit missing: {name}")
        result[name] = RAW_COLUMN_UNITS[name]
    for name in derived_columns:
        result[name] = "dimensionless" if name in {
            "ENDPOINT_DIRECTION_COS",
            "ENDPOINT_DIRECTION_SIN",
        } else "m/s"
    return result


def _validate_code_config_runtime_identities(
    value: Any,
    *,
    amendment: Mapping[str, Any],
    amendment_identity: Mapping[str, Any],
    amendment_v4_identity: Mapping[str, Any],
    code_identities: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(value, dict), "code_config_runtime_identities object required")
    _exact_keys(value, CODE_CONFIG_RUNTIME_KEYS, "code_config_runtime_identities")
    _exact_keys(amendment_identity, AMENDMENT_IDENTITY_KEYS, "amendment identity")
    _exact_keys(amendment_v4_identity, AMENDMENT_IDENTITY_KEYS, "amendment_v4 identity")
    _exact_keys(code_identities, CODE_IDENTITY_KEYS, "code identity roles")
    for role, record in code_identities.items():
        _validate_identity_record(record, f"code identity {role}")
    expected = {
        "amendment": dict(amendment_identity),
        "amendment_v4": dict(amendment_v4_identity),
        "code_identities": dict(code_identities),
        "runtime_identity": amendment["runtime_identity"],
    }
    _require(value == expected, "code/config/runtime identity crosslink mismatch")
    return dict(value)


def validate_family_lock_payload(
    payload: Mapping[str, Any],
    *,
    root: Path,
    amendment: Mapping[str, Any],
    amendment_identity: Mapping[str, Any],
    amendment_v4_identity: Mapping[str, Any],
    decision_payload: Mapping[str, Any],
    code_identities: Mapping[str, Any],
    postrun_pass_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact future LOCK/tombstone payload without publishing it."""

    _require(isinstance(payload, dict), "family lock object required")
    _exact_keys(payload, LOCK_TOP_KEYS, "family lock")
    _require(type(payload["schema_version"]) is int and payload["schema_version"] == 3, "family lock schema_version mismatch")
    _require(payload["artifact_type"] == LOCK_ARTIFACT_TYPE, "family lock artifact_type mismatch")
    _validate_decision(decision_payload, amendment)
    decision_statuses = amendment["output_contract"]["decision_status_literals_exact"]
    lock_statuses = amendment["output_contract"]["family_lock_status_literals_exact"]
    status_role = next(
        (role for role, value in decision_statuses.items() if value == decision_payload["status"]),
        None,
    )
    _require(status_role is not None, "family lock decision status is unfrozen")
    _require(payload["status"] == lock_statuses[status_role], "family lock terminal status mismatch")

    selected = decision_payload["selected_family"]
    if status_role == "positive":
        raw_columns = list(decision_payload["selected_raw_columns"])
        derived_columns = list(decision_payload["selected_derived_columns"])
        _require(payload["selected_family"] == selected, "positive family lock selection mismatch")
        _require(payload["ordered_raw_columns"] == raw_columns, "positive family lock raw order mismatch")
        _require(payload["ordered_derived_columns"] == derived_columns, "positive family lock derived order mismatch")
        _require(payload["units"] == _selected_units(raw_columns, derived_columns), "positive family lock units mismatch")
        _require(
            payload["grib_selectors"]
            == {name: GRIB_SELECTOR_BY_RAW_COLUMN[name] for name in raw_columns},
            "positive family lock GRIB selectors mismatch",
        )
    else:
        _require(
            all(
                payload[field] is None
                for field in (
                    "selected_family",
                    "ordered_raw_columns",
                    "ordered_derived_columns",
                    "units",
                    "grib_selectors",
                )
            ),
            "negative/ambiguity family lock null5 mismatch",
        )

    site_group_scope, spatial_transforms = _lock_static_sections(root, amendment)
    _require(payload["site_group_scope"] == site_group_scope, "family lock site/group scope mismatch")
    _require(payload["spatial_transforms"] == spatial_transforms, "family lock spatial transforms mismatch")
    _require(
        payload["parent_and_independent_gate_results"] == decision_payload,
        "family lock staged decision exact16 gate crosslink mismatch",
    )
    _require(payload["bound_input_identities"] == amendment["bound_inputs"], "family lock bound input exact31 mismatch")
    _validate_code_config_runtime_identities(
        payload["code_config_runtime_identities"],
        amendment=amendment,
        amendment_identity=amendment_identity,
        amendment_v4_identity=amendment_v4_identity,
        code_identities=code_identities,
    )
    _validate_identity_record(postrun_pass_identity, "postrun pass identity")
    _require(payload["postrun_pass_identity"] == postrun_pass_identity, "family lock postrun identity mismatch")
    expected_attestation = decision_payload[
        "no_label_network_future_year_model_submission_attestation"
    ]
    _require(expected_attestation == NO_FORBIDDEN_ACCESS_ATTESTATION, "decision forbidden-access attestation drift")
    _require(payload["no_label_attestation"] == expected_attestation, "family lock no-label attestation mismatch")
    frozen_downstream = amendment["output_contract"]["output_record_schemas_exact"][
        "TARGET_FREE_FAMILY_LOCK_json"
    ]["downstream_authority_literal"]
    _require(
        payload["downstream_authority"] == frozen_downstream == LOCK_DOWNSTREAM_AUTHORITY,
        "family lock downstream authority mismatch",
    )
    return dict(payload)


def _validate_output_identity_record(
    record: Any,
    *,
    basename: str,
) -> dict[str, Any]:
    _require(isinstance(record, dict), f"manifest output identity object required: {basename}")
    _exact_keys(record, OUTPUT_IDENTITY_KEYS, f"manifest output identity {basename}")
    _require(record["path"] == f"{OUTPUT_ROOT_REL}/{basename}", f"manifest output path mismatch: {basename}")
    _require(type(record["size_bytes"]) is int and record["size_bytes"] > 0, f"manifest output size mismatch: {basename}")
    _require(isinstance(record["sha256"], str) and SHA256_RE.fullmatch(record["sha256"]) is not None, f"manifest output sha256 mismatch: {basename}")
    _require(record["format"] == FINAL_OUTPUT_FORMATS[basename], f"manifest output format mismatch: {basename}")
    _require(type(record["row_count"]) is int and record["row_count"] == FINAL_OUTPUT_ROWS[basename], f"manifest output rows mismatch: {basename}")
    _require(isinstance(record["logical_sha256"], str) and SHA256_RE.fullmatch(record["logical_sha256"]) is not None, f"manifest output logical sha256 mismatch: {basename}")
    if record["format"] in {"CSV", "JSON"}:
        _require(record["logical_sha256"] == record["sha256"], f"manifest text logical identity mismatch: {basename}")
    return dict(record)


def validate_run_manifest_payload(
    payload: Mapping[str, Any],
    *,
    amendment: Mapping[str, Any],
    amendment_identity: Mapping[str, Any],
    amendment_v4_identity: Mapping[str, Any],
    decision_payload: Mapping[str, Any],
    code_identities: Mapping[str, Any],
    control_identities: Mapping[str, Any],
    alias_identities: Mapping[str, Any],
    output_identities: Mapping[str, Any],
    fit_ledger: Mapping[str, Any],
    expected_oof_slot_counts: Mapping[str, Any],
    threadpool_evidence: Mapping[str, Any],
    postrun_created_utc: str,
) -> dict[str, Any]:
    """Validate the exact non-circular future manifest payload in memory."""

    _require(isinstance(payload, dict), "run manifest object required")
    _exact_keys(payload, MANIFEST_TOP_KEYS, "run manifest")
    _require(type(payload["schema_version"]) is int and payload["schema_version"] == 3, "run manifest schema_version mismatch")
    _require(payload["artifact_type"] == MANIFEST_ARTIFACT_TYPE, "run manifest artifact_type mismatch")
    _require(payload["status"] == MANIFEST_STATUS, "run manifest status mismatch")
    _validate_decision(decision_payload, amendment)
    _require(payload["terminal_state"] == decision_payload["status"], "run manifest terminal_state mismatch")
    postrun_time = _parse_utc_microseconds(postrun_created_utc, "postrun.created_utc")
    _require(
        _parse_utc_microseconds(AMENDMENT_V4_CREATED_UTC, "amendment_v4.created_utc")
        < postrun_time
        < _parse_utc_microseconds(payload["created_utc"], "run manifest.created_utc"),
        "run manifest chronology mismatch",
    )
    _validate_code_config_runtime_identities(
        payload["code_config_runtime_identities"],
        amendment=amendment,
        amendment_identity=amendment_identity,
        amendment_v4_identity=amendment_v4_identity,
        code_identities=code_identities,
    )

    _require(isinstance(control_identities, dict), "expected control identities object required")
    _exact_keys(control_identities, CONTROL_IDENTITY_KEYS, "expected control identities")
    _require(payload["control_identities"] == control_identities, "run manifest control identities mismatch")
    for role, record in control_identities.items():
        _validate_identity_record(record, f"run manifest control identity {role}")

    _require(isinstance(alias_identities, dict), "expected alias identities object required")
    _exact_keys(alias_identities, ALIAS_SPECS, "expected alias identities")
    _require(payload["alias_identities"] == alias_identities, "run manifest alias identities mismatch")
    for name, record in alias_identities.items():
        _validate_identity_record(record, f"run manifest alias identity {name}")
        _require(record["path"] == f"{OUTPUT_ROOT_REL}/{name}", f"run manifest alias path mismatch: {name}")

    _require(isinstance(output_identities, dict), "expected output identities object required")
    _exact_keys(output_identities, FINAL_OUTPUT_ORDER, "expected output identities")
    _require(payload["output_identities"] == output_identities, "run manifest output identities mismatch")
    for basename in FINAL_OUTPUT_ORDER:
        _validate_output_identity_record(output_identities[basename], basename=basename)
    _require(
        "TRACK_A_TARGET_FREE_RUN_MANIFEST.json" not in output_identities,
        "run manifest must not self-bind its unpublished identity",
    )

    fit_schema = amendment["output_contract"]["output_record_schemas_exact"][
        "TARGET_FREE_FIT_LEDGER_V3_json"
    ]
    _exact_keys(fit_ledger, fit_schema["top_keys_exact"], "manifest source fit ledger")
    _require(
        fit_ledger["schema_version"] == 3
        and fit_ledger["artifact_type"] == "TARGET_FREE_FIT_LEDGER_V3"
        and fit_ledger["status"] == decision_payload["status"],
        "manifest source fit-ledger envelope mismatch",
    )
    _validate_fit_ledger(fit_ledger, amendment, expected_rows=1_332)
    expected_fit_counts = {key: fit_ledger[key] for key in FIT_COUNT_TREE_KEYS}
    _exact_keys(payload["planned_completed_skipped_fit_counts"], FIT_COUNT_TREE_KEYS, "run manifest fit counts")
    _require(
        payload["planned_completed_skipped_fit_counts"] == expected_fit_counts,
        "run manifest fit-ledger count tree mismatch",
    )
    _exact_keys(expected_oof_slot_counts, OOF_SLOT_COUNT_KEYS, "expected OOF slot counts")
    _require(payload["oof_slot_counts"] == expected_oof_slot_counts, "run manifest OOF slot counts mismatch")
    for key, value in expected_oof_slot_counts.items():
        _require(type(value) is int and value >= 0, f"run manifest OOF count invalid: {key}")
    _require(
        expected_oof_slot_counts["completed_oof_slots"]
        + expected_oof_slot_counts["skipped_oof_slots"]
        == expected_oof_slot_counts["planned_oof_slots"]
        == 352_512,
        "run manifest OOF accounting equation mismatch",
    )
    _require(
        expected_oof_slot_counts["independent_selected_oof_slots"]
        <= expected_oof_slot_counts["completed_oof_slots"],
        "run manifest selected OOF count exceeds completed slots",
    )

    _require(isinstance(threadpool_evidence, dict), "expected threadpool evidence object required")
    _exact_keys(threadpool_evidence, THREADPOOL_EVIDENCE_KEYS, "expected threadpool evidence")
    _require(
        payload["threadpool_info_before_inside_after"] == threadpool_evidence,
        "run manifest threadpool evidence mismatch",
    )
    _exact_keys(
        payload["no_forbidden_access_attestation"],
        NO_FORBIDDEN_ACCESS_ATTESTATION,
        "run manifest forbidden-access attestation",
    )
    _require(
        payload["no_forbidden_access_attestation"]
        == decision_payload["no_label_network_future_year_model_submission_attestation"]
        == NO_FORBIDDEN_ACCESS_ATTESTATION,
        "run manifest forbidden-access attestation mismatch",
    )
    _require(payload["manifest_is_last_commit_marker"] is True, "run manifest last-commit marker false")
    return dict(payload)


def _metric_values(np: Any, actual: Any, prediction: Any) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    _require(actual.ndim == prediction.ndim == 1 and actual.size == prediction.size > 0, "metric vectors invalid")
    _require(bool(np.all(np.isfinite(actual))) and bool(np.all(np.isfinite(prediction))), "metric vectors non-finite")
    residual = np.subtract(actual, prediction, dtype=np.float64)
    sse = np.sum(np.multiply(residual, residual, dtype=np.float64), dtype=np.float64)
    mean = np.mean(actual, dtype=np.float64)
    centered = np.subtract(actual, mean, dtype=np.float64)
    sst = np.sum(np.multiply(centered, centered, dtype=np.float64), dtype=np.float64)
    rmse = np.sqrt(np.divide(sse, np.float64(actual.size)), dtype=np.float64)
    q05 = np.quantile(actual, np.float64(0.05), method="linear")
    q95 = np.quantile(actual, np.float64(0.95), method="linear")
    denominator = np.maximum(np.subtract(q95, q05, dtype=np.float64), np.float64(1e-9))
    nrmse = np.divide(rmse, denominator, dtype=np.float64)
    r2 = np.subtract(np.float64(1.0), np.divide(sse, sst, dtype=np.float64), dtype=np.float64)
    return {
        "sse": float(sse),
        "rmse": float(rmse),
        "r2": float(r2),
        "nrmse": float(nrmse),
        "actual_q05": float(q05),
        "actual_q95": float(q95),
        "nrmse_denominator": float(denominator),
        "sst": float(sst),
    }


def _require_same_float(observed: float | None, expected: float, label: str) -> None:
    _require(observed is not None and _float_bits_equal(observed, expected), f"{label}: float mismatch")


def _validate_metrics_csv(
    np: Any,
    rows: list[dict[str, str]],
    oof_table: Any,
    amendment: Mapping[str, Any],
    *,
    production_shape: bool,
) -> dict[str, str | None]:
    metric_names = (
        "sse",
        "rmse",
        "r2",
        "nrmse",
        "actual_q05",
        "actual_q95",
        "nrmse_denominator",
        "pearson",
        "spearman",
        "affine_oof_nrmse",
    )
    parsed: list[dict[str, Any]] = []
    invalid_reasons = frozenset(amendment["metric_primitives"]["invalid_or_skip_reason_enum_exact"])
    for ordinal, row in enumerate(rows):
        label = f"metrics row {ordinal}"
        valid = _csv_bool(row["valid"], label)
        _csv_int(row["rows"], label)
        numeric = {name: _csv_float_pair(row, name, label=label) for name in metric_names}
        if valid:
            _require(row["invalid_reason"] == "", f"{label}: valid row has invalid reason")
        else:
            _require(row["invalid_reason"] in invalid_reasons, f"{label}: invalid reason outside frozen enum")
        _csv_bool(row["independent_selected"], f"{label}.independent_selected")
        _csv_bool(row["parent_max_r2_witness"], f"{label}.parent_max_r2_witness")
        _csv_bool(
            row["parent_redundancy_boolean"],
            f"{label}.parent_redundancy_boolean",
            nullable=True,
        )
        _csv_bool(row["affine_qualifies"], f"{label}.affine_qualifies")
        _csv_boundary_flags(row["boundary_equality_flags"], f"{label}: boundary flags")
        parsed.append({"row": row, "valid": valid, **numeric})

    if not production_shape:
        return {}
    response_order = amendment["planned_fit_budget"]["response_order"]
    predictor_order = amendment["input_and_join_contract"]["predictor_columns_exact"]
    estimators = ("RIDGE_PIPELINE", "EXTRA_TREES")
    _require(len(rows) == len(response_order) * 37, "metrics production row count formula mismatch")
    record_counts: dict[str, int] = {}
    for row in rows:
        record_counts[row["record_kind"]] = record_counts.get(row["record_kind"], 0) + 1
    _require(
        record_counts
        == amendment["output_contract"]["output_record_schemas_exact"]["TARGET_FREE_DUPLICATE_METRICS_csv"]["record_kind_counts"],
        "metrics record kind counts mismatch",
    )

    oof = {name: oof_table.column(name).to_numpy(zero_copy_only=False) for name in oof_table.schema.names}
    selected_by_response: dict[str, str | None] = {}
    for response_index, response in enumerate(response_order):
        base = response_index * 37
        pooled = rows[base : base + 2]
        affine = rows[base + 2 : base + 37]
        for estimator_index, estimator in enumerate(estimators):
            row = pooled[estimator_index]
            _require(row["record_kind"] == "POOLED_ESTIMATOR", "metrics pooled row kind/order mismatch")
            _require(row["response"] == response and row["estimator"] == estimator, "metrics pooled response/estimator order mismatch")
            _require(row["predictor"] == "", "metrics pooled predictor must be empty")
            _require(_csv_int(row["rows"], "metrics pooled rows") == 19_584, "metrics pooled rows mismatch")
            start = (response_index * 2 + estimator_index) * 19_584
            stop = start + 19_584
            status = oof["slot_status"][start:stop]
            if bool(np.all(status == "COMPLETED")):
                _require(parsed[base + estimator_index]["valid"] is True, "completed pooled metric marked invalid")
                metrics = _metric_values(np, oof["actual"][start:stop], oof["oof_prediction"][start:stop])
                for name in ("sse", "rmse", "r2", "nrmse", "actual_q05", "actual_q95", "nrmse_denominator"):
                    _require_same_float(parsed[base + estimator_index][name], metrics[name], f"metrics {response} {estimator} {name}")
            else:
                _require(parsed[base + estimator_index]["valid"] is False, "skipped pooled metric marked valid")
        for predictor_index, predictor in enumerate(predictor_order):
            row = affine[predictor_index]
            _require(row["record_kind"] == "AFFINE_PREDICTOR", "metrics affine row kind/order mismatch")
            _require(row["response"] == response and row["predictor"] == predictor, "metrics affine predictor order mismatch")
            _require(row["estimator"] == "", "metrics affine estimator must be empty")
            if _csv_bool(row["valid"], "metrics affine valid"):
                pearson = parsed[base + 2 + predictor_index]["pearson"]
                spearman = parsed[base + 2 + predictor_index]["spearman"]
                affine_nrmse = parsed[base + 2 + predictor_index]["affine_oof_nrmse"]
                _require(pearson is not None and spearman is not None and affine_nrmse is not None, "valid affine metrics missing")
                equality = abs(pearson) == 0.9999 or abs(spearman) == 0.9999 or affine_nrmse == 0.01
                qualifies = abs(pearson) > 0.9999 and abs(spearman) > 0.9999 and affine_nrmse < 0.01
                _require(_csv_bool(row["affine_qualifies"], "metrics affine qualifies") is qualifies, "affine qualifies mismatch")
                if equality:
                    _require(row["boundary_equality_flags"] != "", "affine cutoff equality lacks flag")

        ridge = parsed[base]
        extra = parsed[base + 1]
        if ridge["valid"] and extra["valid"]:
            ridge_rmse = ridge["rmse"]
            extra_rmse = extra["rmse"]
            assert ridge_rmse is not None and extra_rmse is not None
            if ridge_rmse < extra_rmse:
                selected = "RIDGE_PIPELINE"
            elif extra_rmse < ridge_rmse:
                selected = "EXTRA_TREES"
            else:
                selected = None
            selected_by_response[response] = selected
            for estimator_index, estimator in enumerate(estimators):
                observed = _csv_bool(pooled[estimator_index]["independent_selected"], "pooled independent_selected", nullable=True)
                _require(observed is (estimator == selected) if selected is not None else observed is False, "independent selected mismatch")
        else:
            selected_by_response[response] = None
    return selected_by_response


def _validate_stability_csv(
    np: Any,
    rows: list[dict[str, str]],
    amendment: Mapping[str, Any],
    *,
    production_shape: bool,
    site_to_group: Mapping[str, str] | None = None,
) -> dict[str, dict[str, bool]]:
    parsed: list[dict[str, Any]] = []
    invalid_reasons = frozenset(amendment["metric_primitives"]["invalid_or_skip_reason_enum_exact"])
    for ordinal, row in enumerate(rows):
        label = f"stability row {ordinal}"
        valid = _csv_bool(row["valid"], label)
        count = _csv_int(row["rows"], label)
        _require(count >= 0, f"{label}: row count invalid")
        unique = _csv_int(row["unique_count"], label) if row["unique_count"] else None
        values = {
            name: _csv_float_pair(row, name, label=label)
            for name in ("sst", "sse", "rmse", "r2", "nrmse", "pooled_nrmse_denominator")
        }
        if valid:
            _require(row["invalid_reason"] == "", f"{label}: valid row has reason")
            _require(unique is not None and unique >= 3, f"{label}: valid row unique count invalid")
            _require(all(value is not None for value in values.values()), f"{label}: valid metrics missing")
            sst = values["sst"]
            sse = values["sse"]
            rmse = values["rmse"]
            r2 = values["r2"]
            nrmse = values["nrmse"]
            denominator = values["pooled_nrmse_denominator"]
            assert None not in (sst, sse, rmse, r2, nrmse, denominator)
            _require_same_float(rmse, float(np.sqrt(np.divide(np.float64(sse), np.float64(count)))), f"{label}.rmse")
            _require_same_float(r2, float(np.subtract(np.float64(1), np.divide(np.float64(sse), np.float64(sst)))), f"{label}.r2")
            _require_same_float(nrmse, float(np.divide(np.float64(rmse), np.float64(denominator))), f"{label}.nrmse")
            weak_applicable = _csv_bool(row["weak_floor_applicable"], label)
            equal_r2 = r2 == 0.995
            equal_nrmse = nrmse == 0.02
            equal_weak = weak_applicable and nrmse == 0.01
            _require(_csv_bool(row["r2_boundary_equal_0_995"], label) is equal_r2, f"{label}: r2 equality flag mismatch")
            _require(_csv_bool(row["nrmse_boundary_equal_0_02"], label) is equal_nrmse, f"{label}: nrmse equality flag mismatch")
            _require(_csv_bool(row["nrmse_boundary_equal_0_01"], label) is equal_weak, f"{label}: weak equality flag mismatch")
            strict_pass = r2 < 0.995 or nrmse > 0.02
            _require(_csv_bool(row["strict_not_exact_pass"], label) is strict_pass, f"{label}: strict pass mismatch")
            weak_pass = _csv_bool(row["weak_floor_pass"], label, nullable=True)
            if weak_applicable:
                _require(weak_pass is (nrmse > 0.01), f"{label}: weak floor pass mismatch")
            else:
                _require(weak_pass is None, f"{label}: inapplicable weak floor must be null")
            boundary = equal_r2 or equal_nrmse or equal_weak
            expected_verdict = (
                "AMBIGUOUS_THRESHOLD_EQUALITY"
                if boundary
                else "PASS_STRICT_INTERIOR"
                if strict_pass and (not weak_applicable or bool(weak_pass))
                else "CLEAR_FAIL_STRICT_INTERIOR"
            )
        else:
            _require(row["invalid_reason"] in invalid_reasons, f"{label}: invalid reason outside frozen enum")
            _require(all(values[name] is None for name in ("sst", "sse", "rmse", "r2", "nrmse")), f"{label}: invalid cell metrics must be null")
            _require(values["pooled_nrmse_denominator"] is not None, f"{label}: invalid cell pooled denominator missing")
            weak_applicable = _csv_bool(row["weak_floor_applicable"], label)
            weak_pass = _csv_bool(row["weak_floor_pass"], label, nullable=True)
            _require(weak_pass is None, f"{label}: invalid weak floor pass must be null")
            equal_r2 = _csv_bool(row["r2_boundary_equal_0_995"], label)
            equal_nrmse = _csv_bool(row["nrmse_boundary_equal_0_02"], label)
            equal_weak = _csv_bool(row["nrmse_boundary_equal_0_01"], label)
            strict_pass = _csv_bool(row["strict_not_exact_pass"], label)
            _require(not any((equal_r2, equal_nrmse, equal_weak, strict_pass)), f"{label}: invalid cell flags must be false")
            boundary = False
            expected_verdict = "UNUSABLE"
        _require(row["cell_verdict"] == expected_verdict, f"{label}: cell verdict arithmetic mismatch")
        parsed.append(
            {
                "row": row,
                "valid": valid,
                "strict": strict_pass,
                "weak_applicable": weak_applicable,
                "weak_pass": weak_pass,
                "r2_boundary": equal_r2,
                "nrmse_boundary": equal_nrmse,
                "weak_boundary": equal_weak,
                "boundary": boundary,
            }
        )
    evidence: dict[str, dict[str, bool]] = {}
    if production_shape:
        diagnostics = amendment["planned_fit_budget"]["response_order"] + amendment["output_contract"]["output_record_schemas_exact"]["TARGET_FREE_DISTRIBUTION_STABILITY_V3_json"]["endpoint_pooled_diagnostic_order"]
        _require(len(rows) == len(diagnostics) * 29, "stability production row formula mismatch")
        expected_scope_sequence = ["YEAR"] * 2 + ["SEASON"] * 4 + ["FORECAST_HOUR"] * 3 + ["GROUP"] * 3 + ["SITE"] * 17
        expected_labels = ["2022", "2023", "DJF", "MAM", "JJA", "SON", "EARLY", "MIDDLE", "LATE", "kpx_group_1", "kpx_group_2", "kpx_group_3", *(f"{site:02d}" for site in range(1, 18))]
        expected_rows = [9792] * 2 + [4896] * 4 + [6528] * 3 + [1152] * 3 + [1152] * 17
        _require(site_to_group is not None, "stability production site/group map required")
        _exact_keys(site_to_group, tuple(f"{site:02d}" for site in range(1, 18)), "stability site/group map")
        _require(
            {group: sum(value == group for value in site_to_group.values()) for group in ("kpx_group_1", "kpx_group_2", "kpx_group_3")}
            == {"kpx_group_1": 6, "kpx_group_2": 6, "kpx_group_3": 5},
            "stability site/group membership counts mismatch",
        )
        for diagnostic_index, diagnostic in enumerate(diagnostics):
            block = parsed[diagnostic_index * 29 : (diagnostic_index + 1) * 29]
            raw_rows = [item["row"] for item in block]
            _require(all(row["response_or_endpoint"] == diagnostic for row in raw_rows), "stability diagnostic order mismatch")
            _require([row["diagnostic_kind"] for row in raw_rows] == (["RAW_RESPONSE"] if diagnostic in amendment["planned_fit_budget"]["response_order"] else ["ENDPOINT_VETO"]) * 29, "stability diagnostic kind mismatch")
            _require([row["scope_type"] for row in raw_rows] == expected_scope_sequence, "stability scope order mismatch")
            _require([row["scope_label"] for row in raw_rows] == expected_labels, "stability scope label order mismatch")
            _require([_csv_int(row["rows"], "stability production rows") for row in raw_rows] == expected_rows, "stability cell row counts mismatch")

            year_pass = all(item["valid"] and item["strict"] and not item["r2_boundary"] and not item["nrmse_boundary"] for item in block[0:2])
            season = block[2:6]
            season_strict_count = sum(item["strict"] for item in season)
            season_boundary = any(item["boundary"] for item in season)
            if season_strict_count == 4:
                season_pass = not season_boundary
                _require(not any(item["weak_applicable"] for item in season), f"stability unexpected season weak floor: {diagnostic}")
            elif season_strict_count == 3:
                remaining = [item for item in season if not item["strict"]]
                _require(len(remaining) == 1 and sum(item["weak_applicable"] for item in season) == 1 and remaining[0]["weak_applicable"], f"stability season weak-floor routing mismatch: {diagnostic}")
                season_pass = bool(remaining[0]["valid"] and remaining[0]["weak_pass"] and not season_boundary)
            else:
                season_pass = False
                _require(not any(item["weak_applicable"] for item in season), f"stability unexpected season weak floor: {diagnostic}")
            forecast_pass = all(item["valid"] and item["strict"] and item["weak_applicable"] and bool(item["weak_pass"]) and not item["boundary"] for item in block[6:9])
            _require(all(item["weak_applicable"] for item in block[6:9]), f"stability forecast weak-floor applicability mismatch: {diagnostic}")
            group_pass = all(item["valid"] and item["strict"] and not item["boundary"] for item in block[9:12])
            site_pass_counts = {"kpx_group_1": 0, "kpx_group_2": 0, "kpx_group_3": 0}
            for site_offset, item in enumerate(block[12:29], 1):
                if item["valid"] and item["strict"] and not item["boundary"]:
                    site_pass_counts[site_to_group[f"{site_offset:02d}"]] += 1
            site_pass = all(site_pass_counts[group] >= required for group, required in {"kpx_group_1": 5, "kpx_group_2": 5, "kpx_group_3": 4}.items())
            evidence[diagnostic] = {
                "pass": all((year_pass, season_pass, forecast_pass, group_pass, site_pass)),
                "boundary": any(item["boundary"] for item in block),
                "available": any(item["valid"] for item in block),
            }
    return evidence


def _validate_endpoint_pooled_metrics(
    np: Any,
    distribution: Mapping[str, Any],
    derived_table: Any,
    amendment: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    derived = {name: derived_table.column(name).to_numpy(zero_copy_only=False) for name in derived_table.schema.names}
    endpoints = amendment["output_contract"]["output_record_schemas_exact"]["TARGET_FREE_DISTRIBUTION_STABILITY_V3_json"]["endpoint_pooled_diagnostic_order"]
    records = distribution["endpoint_pooled_metric_records"]
    invalid_reasons = frozenset(amendment["metric_primitives"]["invalid_or_skip_reason_enum_exact"])
    evidence: dict[str, dict[str, Any]] = {}
    for ordinal, diagnostic in enumerate(endpoints):
        record = records[ordinal]
        _require(record["diagnostic"] == diagnostic, "endpoint pooled order mismatch")
        site_mask = (derived["diagnostic"] == diagnostic) & (derived["scope_type"] == "SITE")
        _require(int(np.count_nonzero(site_mask)) == 19_584, f"endpoint pooled SITE rows mismatch: {diagnostic}")
        valid_mask = site_mask & derived["valid"].astype(bool)
        binding = record["source_component_estimator_binding"]
        _require(isinstance(binding, dict), f"endpoint {diagnostic}: source binding object required")
        _exact_keys(binding, _endpoint_source_responses(diagnostic), f"endpoint pooled binding {diagnostic}")
        _require(all(value in (*ESTIMATOR_ORDER, None) for value in binding.values()), f"endpoint {diagnostic}: source binding estimator invalid")
        if record["valid"] is True:
            _require(int(np.count_nonzero(valid_mask)) == 19_584, f"endpoint pooled valid rows mismatch: {diagnostic}")
            metrics = _metric_values(np, derived["actual"][valid_mask], derived["oof_prediction"][valid_mask])
            _require(record["rows"] == 19_584 and record["invalid_reason"] is None, f"endpoint pooled validity mismatch: {diagnostic}")
            for name in ("sse", "rmse", "r2", "nrmse", "actual_q05", "actual_q95", "nrmse_denominator"):
                observed = _json_float_pair(record, name, label=f"endpoint {diagnostic}")
                _require_same_float(observed, metrics[name], f"endpoint {diagnostic}.{name}")
            _require(record["r2_boundary_equal_0_9925"] is (metrics["r2"] == 0.9925), f"endpoint {diagnostic}: r2 boundary mismatch")
            _require(record["nrmse_boundary_equal_0_025"] is (metrics["nrmse"] == 0.025), f"endpoint {diagnostic}: nrmse boundary mismatch")
            _require(record["strict_stable_margin_pass"] is (metrics["r2"] < 0.9925 and metrics["nrmse"] > 0.025), f"endpoint {diagnostic}: stable margin mismatch")
        else:
            _require(int(np.count_nonzero(valid_mask)) == 0, f"endpoint invalid record has valid derived rows: {diagnostic}")
            _require(record["invalid_reason"] in invalid_reasons, f"endpoint invalid reason outside frozen enum: {diagnostic}")
            _require(record["strict_stable_margin_pass"] is False, f"endpoint invalid strict pass must be false: {diagnostic}")
            _require(record["r2_boundary_equal_0_9925"] is False and record["nrmse_boundary_equal_0_025"] is False, f"endpoint invalid boundary must be false: {diagnostic}")
        evidence[diagnostic] = {
            "available": record["valid"],
            "pass": record["strict_stable_margin_pass"],
            "boundary": record["r2_boundary_equal_0_9925"] or record["nrmse_boundary_equal_0_025"],
            "binding": binding,
            "verdict": record["endpoint_pooled_verdict"],
        }
    return evidence


def _validate_staged_decision_crosslinks(
    np: Any,
    decision_payload: Mapping[str, Any],
    metrics_rows: Sequence[Mapping[str, str]],
    oof_table: Any,
    stability_evidence: Mapping[str, Mapping[str, bool]],
    distribution_evidence: Mapping[str, Mapping[str, bool]],
    endpoint_pooled_evidence: Mapping[str, Mapping[str, Any]],
    amendment: Mapping[str, Any],
) -> None:
    responses = tuple(amendment["planned_fit_budget"]["response_order"])
    predictors = tuple(amendment["input_and_join_contract"]["predictor_columns_exact"])
    endpoints = tuple(amendment["derived_wind_contract"]["endpoint_veto_diagnostics_exact"])
    raw_by_response = {record["response"]: record for record in decision_payload["raw_component_results"]}
    endpoint_by_diagnostic = {record["diagnostic"]: record for record in decision_payload["endpoint_veto_results"]}
    oof = {name: oof_table.column(name).to_numpy(zero_copy_only=False) for name in oof_table.schema.names}

    for response_index, response in enumerate(responses):
        raw = raw_by_response[response]
        block = list(metrics_rows[response_index * 37 : (response_index + 1) * 37])
        _require(len(block) == 37, f"metrics/decision block length mismatch: {response}")
        expected_flags = ";".join(raw["boundary_equality_flags"])
        _require(all(row["component_classification"] == raw["classification"] for row in block), f"metrics/decision classification mismatch: {response}")
        _require(all(row["boundary_equality_flags"] == expected_flags for row in block), f"metrics/decision boundary flag mismatch: {response}")
        pooled = block[:2]
        affine = block[2:]
        pooled_valid = [_csv_bool(row["valid"], f"metrics crosslink {response}") for row in pooled]
        _require(pooled_valid[0] is pooled_valid[1], f"metrics pooled validity partition mismatch: {response}")
        affine_hits = [
            predictor
            for predictor, row in zip(predictors, affine, strict=True)
            if _csv_bool(row["affine_qualifies"], f"metrics affine crosslink {response}")
        ]
        _require(raw["affine_qualifying_predictors"] == affine_hits, f"metrics/decision affine witness mismatch: {response}")
        _require(all(_csv_bool(row["independent_selected"], f"metrics affine independent {response}") is False for row in affine), f"affine rows cannot be independently selected: {response}")
        _require(all(_csv_bool(row["parent_max_r2_witness"], f"metrics affine parent {response}") is False for row in affine), f"affine rows cannot be parent witnesses: {response}")
        _require(all(_csv_bool(row["parent_redundancy_boolean"], f"metrics affine parent boolean {response}", nullable=True) is None for row in affine), f"affine rows cannot carry parent booleans: {response}")

        if pooled_valid[0]:
            rmse = [_csv_float_pair(row, "rmse", label=f"metrics crosslink {response}") for row in pooled]
            r2 = [_csv_float_pair(row, "r2", label=f"metrics crosslink {response}") for row in pooled]
            nrmse = [_csv_float_pair(row, "nrmse", label=f"metrics crosslink {response}") for row in pooled]
            assert None not in (*rmse, *r2, *nrmse)
            rmse_tie = rmse[0] == rmse[1]
            selected_index = None if rmse_tie else 0 if rmse[0] < rmse[1] else 1
            selected = None if selected_index is None else ESTIMATOR_ORDER[selected_index]
            max_r2 = max(r2)
            parent_indices = [] if rmse_tie else [index for index, value in enumerate(r2) if value == max_r2]
            parents = [ESTIMATOR_ORDER[index] for index in parent_indices]
            parent_booleans = [r2[index] > 0.98 and nrmse[index] < 0.10 for index in parent_indices]
            _require(raw["independent_estimator"] == selected, f"metrics/decision selector mismatch: {response}")
            _require(raw["parent_reference_estimators"] == parents, f"metrics/decision parent witness mismatch: {response}")
            for estimator_index, row in enumerate(pooled):
                expected_selected = selected_index == estimator_index
                expected_parent = estimator_index in parent_indices
                expected_parent_boolean = parent_booleans[parent_indices.index(estimator_index)] if expected_parent else None
                _require(_csv_bool(row["independent_selected"], f"metrics selected {response}") is expected_selected, f"metrics selected flag mismatch: {response}")
                _require(_csv_bool(row["parent_max_r2_witness"], f"metrics parent {response}") is expected_parent, f"metrics parent witness flag mismatch: {response}")
                _require(_csv_bool(row["parent_redundancy_boolean"], f"metrics parent boolean {response}", nullable=True) is expected_parent_boolean, f"metrics parent boolean mismatch: {response}")
            if rmse_tie:
                expected_classification = "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
            elif raw["boundary_equality_flags"]:
                expected_classification = "AMBIGUOUS_THRESHOLD_EQUALITY"
            elif affine_hits:
                expected_classification = "CLEAR_REDUNDANT_AFFINE"
            elif len(set(parent_booleans)) > 1:
                expected_classification = "AMBIGUOUS_PARENT_ESTIMATOR_TIE"
            elif parent_booleans[0]:
                expected_classification = "CLEAR_REDUNDANT_PARENT_OOF"
            else:
                assert selected_index is not None
                if r2[selected_index] > 0.995 and nrmse[selected_index] < 0.02:
                    expected_classification = "CLEAR_REDUNDANT_INDEPENDENT_EXACT_DUPLICATE"
                else:
                    margin = r2[selected_index] < 0.9925 and nrmse[selected_index] > 0.025
                    stable = stability_evidence[response]["pass"]
                    distribution = distribution_evidence[response]["pass"]
                    expected_classification = (
                        "PASS_NONREDUNDANT_MARGIN"
                        if margin and stable and distribution
                        else "CLEAR_UNSTABLE_FAILURE"
                        if margin
                        else "AMBIGUOUS_GRAY_ZONE"
                    )
            _require(raw["classification"] == expected_classification, f"metrics/gates decision classification mismatch: {response}")
            expected_parent_boolean = (
                None
                if expected_classification in AMBIGUOUS_CLASSIFICATIONS
                or (expected_classification == "CLEAR_REDUNDANT_AFFINE" and len(set(parent_booleans)) > 1)
                else parent_booleans[0]
            )
            _require(raw["parent_redundancy_boolean"] is expected_parent_boolean, f"metrics/decision parent redundancy mismatch: {response}")
        else:
            selected = None
            _require(raw["classification"] in PREMODEL_CLASSIFICATIONS, f"invalid pooled metrics require premodel classification: {response}")
            _require(raw["independent_estimator"] is None and raw["parent_reference_estimators"] == [] and not affine_hits, f"premodel metric witnesses must be empty: {response}")
            _require(all(_csv_bool(row["independent_selected"], f"metrics premodel selected {response}") is False for row in pooled), f"premodel pooled selected flag mismatch: {response}")
            _require(all(_csv_bool(row["parent_max_r2_witness"], f"metrics premodel parent {response}") is False for row in pooled), f"premodel pooled parent flag mismatch: {response}")

        classification = raw["classification"]
        unresolved = classification in PREMODEL_CLASSIFICATIONS or classification in AMBIGUOUS_CLASSIFICATIONS
        expected_cell = None if unresolved else stability_evidence[response]["pass"]
        expected_distribution = None if unresolved else distribution_evidence[response]["pass"]
        _require(raw["cell_stability_pass"] is expected_cell, f"stability/decision gate mismatch: {response}")
        _require(raw["distribution_stability_pass"] is expected_distribution, f"distribution/decision gate mismatch: {response}")

        for estimator_index, estimator in enumerate(ESTIMATOR_ORDER):
            start = (response_index * 2 + estimator_index) * 19_584
            stop = start + 19_584
            expected_selected = estimator == selected
            selected_flags = oof["independent_selected"][start:stop]
            _require(bool(np.all(selected_flags == expected_selected)), f"OOF/decision selected flag mismatch: {response}/{estimator}")
            if expected_selected:
                _validate_selected_oof_block_completion(
                    np,
                    oof,
                    start,
                    stop,
                    response=response,
                    estimator=estimator,
                )

    endpoint_records = endpoint_pooled_evidence
    for diagnostic in endpoints:
        endpoint = endpoint_by_diagnostic[diagnostic]
        pooled = endpoint_records[diagnostic]
        _require(pooled["binding"] == endpoint["source_component_estimator_binding"], f"endpoint pooled/decision binding mismatch: {diagnostic}")
        source_unavailable = any(value is None for value in endpoint["source_component_estimator_binding"].values())
        if source_unavailable:
            _require(not pooled["available"], f"unavailable endpoint has valid pooled evidence: {diagnostic}")
            expected_pooled = expected_cell = expected_distribution = None
        else:
            _require(pooled["available"], f"available endpoint lacks pooled evidence: {diagnostic}")
            threshold_ambiguity = endpoint["verdict"] == "AMBIGUOUS_THRESHOLD_EQUALITY"
            expected_pooled = None if pooled["boundary"] else pooled["pass"]
            expected_cell = None if stability_evidence[diagnostic]["boundary"] else stability_evidence[diagnostic]["pass"]
            expected_distribution = None if distribution_evidence[diagnostic]["boundary"] else distribution_evidence[diagnostic]["pass"]
            expected_verdict = (
                "AMBIGUOUS_THRESHOLD_EQUALITY"
                if pooled["boundary"] or stability_evidence[diagnostic]["boundary"] or distribution_evidence[diagnostic]["boundary"]
                else "PASS_ENDPOINT_NONREDUNDANT_MARGIN"
                if pooled["pass"] and stability_evidence[diagnostic]["pass"] and distribution_evidence[diagnostic]["pass"]
                else "CLEAR_ENDPOINT_VETO_FAILURE"
            )
            _require(endpoint["verdict"] == expected_verdict, f"endpoint evidence/decision verdict mismatch: {diagnostic}")
            _require(threshold_ambiguity is (expected_verdict == "AMBIGUOUS_THRESHOLD_EQUALITY"), f"endpoint threshold state mismatch: {diagnostic}")
        _require(endpoint["pooled_stable_margin_pass"] is expected_pooled, f"endpoint pooled gate mismatch: {diagnostic}")
        _require(endpoint["cell_stability_pass"] is expected_cell, f"endpoint cell gate mismatch: {diagnostic}")
        _require(endpoint["distribution_stability_pass"] is expected_distribution, f"endpoint distribution gate mismatch: {diagnostic}")
        _require(pooled["verdict"] == endpoint["verdict"], f"endpoint pooled verdict crosslink mismatch: {diagnostic}")


def _validate_evidence_document_headers(
    ledger: Mapping[str, Any],
    distribution: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    _require(ledger["schema_version"] == 3, "fit ledger schema_version mismatch")
    _require(ledger["artifact_type"] == "TARGET_FREE_FIT_LEDGER_V3", "fit ledger artifact_type mismatch")
    _require(distribution["schema_version"] == 3, "distribution schema_version mismatch")
    _require(distribution["artifact_type"] == "TARGET_FREE_DISTRIBUTION_STABILITY_V3", "distribution artifact_type mismatch")
    _require(ledger["status"] == decision["status"], "fit ledger/decision status mismatch")
    _require(distribution["status"] == decision["status"], "distribution/decision status mismatch")


def _validate_derived_binding_crosslinks(
    bindings: Mapping[str, Mapping[str, str | None]],
    decision: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> None:
    order = amendment["derived_wind_contract"]["derived_order_exact"]
    _require(list(bindings) == order, "derived binding diagnostic order mismatch")
    raw = {record["response"]: record for record in decision["raw_component_results"]}
    for diagnostic in order:
        expected = {
            response: raw[response]["independent_estimator"]
            for response in _derived_source_responses(diagnostic)
        }
        _require(bindings[diagnostic] == expected, f"derived/decision source binding mismatch: {diagnostic}")


def _validate_derived_formula_parity(
    np: Any,
    oof_table: Any,
    derived_table: Any,
    selected: Mapping[str, str | None],
    amendment: Mapping[str, Any],
) -> None:
    """Recompute all valid SITE derived diagnostics from staged raw OOF only."""
    if any(selected.get(name) is None for name in amendment["planned_fit_budget"]["response_order"] if name != "HPBL_surface"):
        return
    oof = {name: oof_table.column(name).to_numpy(zero_copy_only=False) for name in oof_table.schema.names}
    derived = {name: derived_table.column(name).to_numpy(zero_copy_only=False) for name in derived_table.schema.names}
    raw: dict[str, tuple[Any, Any]] = {}
    response_order = amendment["planned_fit_budget"]["response_order"]
    estimator_order = ("RIDGE_PIPELINE", "EXTRA_TREES")
    for response_index, response in enumerate(response_order):
        estimator = selected.get(response)
        if estimator is None:
            continue
        estimator_index = estimator_order.index(estimator)
        start = (response_index * 2 + estimator_index) * 19_584
        stop = start + 19_584
        raw[response] = (
            oof["actual"][start:stop].astype(np.float64, copy=False),
            oof["oof_prediction"][start:stop].astype(np.float64, copy=False),
        )
    if len(raw) < 8:
        return

    def build(which: int) -> dict[str, Any]:
        u925, v925 = raw["UGRD_925mb"][which], raw["VGRD_925mb"][which]
        u950, v950 = raw["UGRD_950mb"][which], raw["VGRD_950mb"][which]
        u975, v975 = raw["UGRD_975mb"][which], raw["VGRD_975mb"][which]
        u1000, v1000 = raw["UGRD_1000mb"][which], raw["VGRD_1000mb"][which]
        du_end = np.subtract(u925, u1000, dtype=np.float64)
        dv_end = np.subtract(v925, v1000, dtype=np.float64)
        ws925, ws950 = np.hypot(u925, v925), np.hypot(u950, v950)
        ws975, ws1000 = np.hypot(u975, v975), np.hypot(u1000, v1000)
        du975 = np.subtract(u975, u1000, dtype=np.float64)
        dv975 = np.subtract(v975, v1000, dtype=np.float64)
        du950 = np.subtract(u950, u975, dtype=np.float64)
        dv950 = np.subtract(v950, v975, dtype=np.float64)
        du925 = np.subtract(u925, u950, dtype=np.float64)
        dv925 = np.subtract(v925, v950, dtype=np.float64)
        denom = np.maximum(np.multiply(ws1000, ws925, dtype=np.float64), np.float64(1e-6))
        cos = np.clip(np.divide(u1000 * u925 + v1000 * v925, denom), -1.0, 1.0)
        sin = np.clip(np.divide(u1000 * v925 - v1000 * u925, denom), -1.0, 1.0)
        calm = (ws1000 < 1e-6) | (ws925 < 1e-6)
        cos = cos.astype(np.float64, copy=True)
        sin = sin.astype(np.float64, copy=True)
        cos[calm] = 0.0
        sin[calm] = 0.0
        return {
            "WS_925": ws925,
            "WS_950": ws950,
            "WS_975": ws975,
            "WS_1000": ws1000,
            "ENDPOINT_DU_925_MINUS_1000": du_end,
            "ENDPOINT_DV_925_MINUS_1000": dv_end,
            "ENDPOINT_VECTOR_SHEAR_MAG": np.hypot(du_end, dv_end),
            "ADJACENT_DU_975_MINUS_1000": du975,
            "ADJACENT_DV_975_MINUS_1000": dv975,
            "ADJACENT_SHEAR_MAG_975_MINUS_1000": np.hypot(du975, dv975),
            "ADJACENT_DU_950_MINUS_975": du950,
            "ADJACENT_DV_950_MINUS_975": dv950,
            "ADJACENT_SHEAR_MAG_950_MINUS_975": np.hypot(du950, dv950),
            "ADJACENT_DU_925_MINUS_950": du925,
            "ADJACENT_DV_925_MINUS_950": dv925,
            "ADJACENT_SHEAR_MAG_925_MINUS_950": np.hypot(du925, dv925),
            "ENDPOINT_DIRECTION_COS": cos,
            "ENDPOINT_DIRECTION_SIN": sin,
        }

    actual_expected = build(0)
    prediction_expected = build(1)
    order = amendment["derived_wind_contract"]["derived_order_exact"]
    _require(list(actual_expected) == order, "derived formula order implementation drift")
    for diagnostic_index, diagnostic in enumerate(order):
        start = diagnostic_index * 23_040
        site_slice = slice(start, start + 19_584)
        _require(bool(np.all(derived["diagnostic"][site_slice] == diagnostic)), f"derived diagnostic order mismatch: {diagnostic}")
        _require(bool(np.all(derived["scope_type"][site_slice] == "SITE")), f"derived SITE block mismatch: {diagnostic}")
        valid = derived["valid"][site_slice].astype(bool)
        if bool(np.all(valid)):
            _require(bool(np.array_equal(derived["actual"][site_slice], actual_expected[diagnostic])), f"derived actual formula mismatch: {diagnostic}")
            _require(bool(np.array_equal(derived["oof_prediction"][site_slice], prediction_expected[diagnostic])), f"derived prediction formula mismatch: {diagnostic}")


def _format_name(name: str) -> str:
    if name.endswith(".csv"):
        return "CSV"
    if name.endswith(".json"):
        return "JSON"
    if name.endswith(".parquet"):
        return "PARQUET"
    _fail(f"unknown staged output format: {name}")
    raise AssertionError


def _staged_identity_record(
    identity: Mapping[str, Any], name: str, row_count: int, logical_sha256: str
) -> dict[str, Any]:
    return {
        **dict(identity),
        "format": _format_name(name),
        "row_count": row_count,
        "logical_sha256": logical_sha256,
    }


def validate_staged_outputs(
    context: PredataContext,
    *,
    count_overrides: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate staged diagnostics after the predata context has been sealed."""
    counts = dict(OUTPUT_COUNTS)
    if count_overrides:
        counts.update(count_overrides)
    production_shape = counts == OUTPUT_COUNTS
    aliases, raw_staged = _validate_namespace_structure(context)
    np, pa, pq = _import_arrow_after_predata()
    _validate_imported_runtime(context, pa, pq)
    _validate_source_parquet_metadata(context, pq)
    schemas = context.amendment["output_contract"]["output_record_schemas_exact"]

    metrics_path, metrics_identity = raw_staged["TARGET_FREE_DUPLICATE_METRICS.csv"]
    metrics_rows, metrics_raw = _read_strict_csv(
        metrics_path,
        schemas["TARGET_FREE_DUPLICATE_METRICS_csv"]["columns_exact"],
        counts["metrics_rows"],
        label="TARGET_FREE_DUPLICATE_METRICS.csv",
    )
    stability_path, stability_identity = raw_staged["TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv"]
    stability_rows, stability_raw = _read_strict_csv(
        stability_path,
        schemas["TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE_csv"]["columns_exact"],
        counts["stability_rows"],
        label="TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv",
    )
    decision_path, decision_identity = raw_staged["TARGET_FREE_FAMILY_DECISION.json"]
    decision_payload, decision_raw = _read_strict_output_json(
        decision_path,
        schemas["TARGET_FREE_FAMILY_DECISION_json"]["top_keys_exact"],
        label="TARGET_FREE_FAMILY_DECISION.json",
    )
    ledger_path, ledger_identity = raw_staged["TARGET_FREE_FIT_LEDGER_V3.json"]
    ledger_payload, ledger_raw = _read_strict_output_json(
        ledger_path,
        schemas["TARGET_FREE_FIT_LEDGER_V3_json"]["top_keys_exact"],
        label="TARGET_FREE_FIT_LEDGER_V3.json",
    )
    distribution_path, distribution_identity = raw_staged["TARGET_FREE_DISTRIBUTION_STABILITY_V3.json"]
    distribution_payload, distribution_raw = _read_strict_output_json(
        distribution_path,
        schemas["TARGET_FREE_DISTRIBUTION_STABILITY_V3_json"]["top_keys_exact"],
        label="TARGET_FREE_DISTRIBUTION_STABILITY_V3.json",
    )
    oof_path, oof_identity = raw_staged["TARGET_FREE_DUPLICATE_OOF_V3.parquet"]
    oof_table, oof_logical = _read_staged_parquet(
        pa,
        pq,
        oof_path,
        schemas["TARGET_FREE_DUPLICATE_OOF_V3_parquet"],
        expected_rows=counts["oof_rows"],
        label="TARGET_FREE_DUPLICATE_OOF_V3.parquet",
    )
    _validate_v4_affected_schema(
        pa,
        oof_table.schema,
        "TARGET_FREE_DUPLICATE_OOF_V3.parquet",
        context.amendment_v4,
    )
    derived_path, derived_identity = raw_staged["TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3.parquet"]
    derived_table, derived_logical = _read_staged_parquet(
        pa,
        pq,
        derived_path,
        schemas["TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3_parquet"],
        expected_rows=counts["derived_rows"],
        label="TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3.parquet",
    )
    _validate_v4_affected_schema(
        pa,
        derived_table.schema,
        "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3.parquet",
        context.amendment_v4,
    )

    _validate_oof_table(np, oof_table, context.amendment, production_shape=production_shape)
    derived_bindings = _validate_derived_table(
        np,
        derived_table,
        context.amendment,
        production_shape=production_shape,
    )
    _validate_fit_ledger(ledger_payload, context.amendment, expected_rows=counts["fit_ledger_rows"])
    if production_shape:
        _validate_fit_ledger_oof_crosslinks(np, ledger_payload, oof_table, context.amendment)
    distribution_evidence = _validate_distribution(
        distribution_payload,
        context.amendment,
        expected_records=counts["distribution_records"],
        expected_endpoint=counts["endpoint_pooled_metric_records"],
    )
    selected = _validate_metrics_csv(np, metrics_rows, oof_table, context.amendment, production_shape=production_shape)
    site_to_group = _oof_site_group_map(np, oof_table) if production_shape else None
    stability_evidence = _validate_stability_csv(
        np,
        stability_rows,
        context.amendment,
        production_shape=production_shape,
        site_to_group=site_to_group,
    )
    decision = _validate_decision(decision_payload, context.amendment)
    _validate_evidence_document_headers(ledger_payload, distribution_payload, decision_payload)
    if production_shape:
        endpoint_pooled_evidence = _validate_endpoint_pooled_metrics(np, distribution_payload, derived_table, context.amendment)
        _validate_staged_decision_crosslinks(
            np,
            decision_payload,
            metrics_rows,
            oof_table,
            stability_evidence,
            distribution_evidence,
            endpoint_pooled_evidence,
            context.amendment,
        )
        _derive_manifest_oof_slot_counts(
            np,
            ledger_payload,
            oof_table,
            decision_payload,
        )
        _validate_derived_binding_crosslinks(derived_bindings, decision_payload, context.amendment)
        _validate_derived_formula_parity(np, oof_table, derived_table, selected, context.amendment)

    staged_records = {
        "TARGET_FREE_DUPLICATE_METRICS.csv": _staged_identity_record(metrics_identity, "TARGET_FREE_DUPLICATE_METRICS.csv", len(metrics_rows), sha256_bytes(metrics_raw)),
        "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv": _staged_identity_record(stability_identity, "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv", len(stability_rows), sha256_bytes(stability_raw)),
        "TARGET_FREE_FAMILY_DECISION.json": _staged_identity_record(decision_identity, "TARGET_FREE_FAMILY_DECISION.json", 1, sha256_bytes(decision_raw)),
        "TARGET_FREE_DUPLICATE_OOF_V3.parquet": _staged_identity_record(oof_identity, "TARGET_FREE_DUPLICATE_OOF_V3.parquet", oof_table.num_rows, oof_logical),
        "TARGET_FREE_FIT_LEDGER_V3.json": _staged_identity_record(ledger_identity, "TARGET_FREE_FIT_LEDGER_V3.json", len(ledger_payload["unit_slots"]), sha256_bytes(ledger_raw)),
        "TARGET_FREE_DISTRIBUTION_STABILITY_V3.json": _staged_identity_record(distribution_identity, "TARGET_FREE_DISTRIBUTION_STABILITY_V3.json", len(distribution_payload["records"]), sha256_bytes(distribution_raw)),
        "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3.parquet": _staged_identity_record(derived_identity, "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3.parquet", derived_table.num_rows, derived_logical),
    }
    _require(list(staged_records) == list(RUNNER_OUTPUTS), "staged identity order mismatch")
    return {"aliases": aliases, "runner_staged_outputs": staged_records}, decision, counts


def _validate_actual_auditor_command(
    context: PredataContext, actual_orig_argv: Sequence[str]
) -> None:
    """Bind the live production invocation to the authorized auditor argv."""

    expected = context.authorization["required_commands"]["auditor"]
    _require(type(expected) is list, "authorized auditor command must be a list")
    _require(
        len(expected) == 10
        and expected[1:4]
        == ["-B", "-m", "scripts.audit_noaa_gfs_target_free_duplicate_v3"]
        and expected[4] == "--root"
        and expected[6] == "--authorization"
        and expected[8] == "--independent-go",
        "authorized auditor command shape mismatch",
    )
    for index, label in (
        (0, "interpreter"),
        (5, "root"),
        (7, "authorization"),
        (9, "independent GO"),
    ):
        _require(
            type(expected[index]) is str and Path(expected[index]).is_absolute(),
            f"authorized auditor {label} must be absolute",
        )
    _require(type(actual_orig_argv) is list, "actual sys.orig_argv must be a list")
    _require(
        all(type(token) is str for token in actual_orig_argv),
        "actual sys.orig_argv tokens must be strings",
    )
    _require(
        _json_type_exact_equal(actual_orig_argv, expected),
        "actual auditor command does not exactly match authorization",
    )


def audit(
    root: Path,
    authorization_path: Path,
    independent_go_path: Path,
    *,
    repo: Path = REPO_ROOT,
    actual_orig_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    context = validate_predata_authority(
        root,
        authorization_path,
        independent_go_path,
        repo=repo,
    )
    if actual_orig_argv is not None:
        _validate_actual_auditor_command(context, actual_orig_argv)
    staged_identities, decision, counts = validate_staged_outputs(context)
    payload = {
        "schema_version": 3,
        "artifact_type": ARTIFACT_TYPE,
        "status": PASS_STATUS,
        "attempt_id": context.attempt_id,
        "amendment": context.amendment_identity,
        "amendment_v4": context.amendment_v4_identity,
        "authorization": context.authorization_identity,
        "independent_go": context.independent_go_identity,
        "code_identities": context.code_seal["code_identities"],
        "staged_output_identities": staged_identities,
        "output_counts": counts,
        "decision": decision,
        "poststate": dict(POSTSTATE),
        "files_written": 0,
        "network_requests": 0,
        "models_fit": 0,
        "forbidden_reads": {
            "labels": False,
            "arrays_2024_2025": False,
            "source_parquet_logical_values": False,
        },
    }
    _exact_keys(payload, AUDITOR_STDOUT_TOP_KEYS, "auditor stdout")
    return payload


_GUARD_INSTALLED = False


def install_readonly_runtime_guard() -> None:
    """Deny writes, process creation, and network for the live auditor process."""
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return
    sys.dont_write_bytecode = True
    write_flags = 0
    for name in ("O_WRONLY", "O_RDWR", "O_APPEND", "O_CREAT", "O_TRUNC", "O_EXCL"):
        write_flags |= int(getattr(os, name, 0))
    forbidden_events = {
        "os.remove",
        "os.rmdir",
        "os.rename",
        "os.replace",
        "os.mkdir",
        "os.link",
        "os.symlink",
        "os.chmod",
        "os.chown",
        "shutil.copyfile",
        "shutil.copymode",
        "shutil.copystat",
        "subprocess.Popen",
    }

    def guard(event: str, args: tuple[Any, ...]) -> None:
        if event.startswith("socket."):
            raise PermissionError("V3 auditor network access denied")
        if event in forbidden_events:
            raise PermissionError(f"V3 auditor mutation denied: {event}")
        if event == "open":
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            if isinstance(mode, str) and any(token in mode for token in ("w", "a", "x", "+")):
                raise PermissionError("V3 auditor write open denied")
            if isinstance(flags, int) and flags & write_flags:
                raise PermissionError("V3 auditor write flags denied")

    sys.addaudithook(guard)
    _GUARD_INSTALLED = True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--independent-go", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    actual_orig_argv_for_test: list[str] | None = None,
) -> int:
    """Run the auditor; explicit ``argv`` is the test-injection boundary."""

    production_invocation = argv is None
    args = _parser().parse_args(argv)
    actual_orig_argv = (
        getattr(sys, "orig_argv", [])
        if production_invocation
        else actual_orig_argv_for_test
    )
    install_readonly_runtime_guard()
    try:
        payload = audit(
            args.root,
            args.authorization,
            args.independent_go,
            actual_orig_argv=actual_orig_argv,
        )
    except Exception as exc:
        failure = {
            "schema_version": 3,
            "artifact_type": ARTIFACT_TYPE,
            "status": FAIL_STATUS,
            "reason": str(exc),
            "files_written": 0,
            "network_requests": 0,
            "models_fit": 0,
        }
        sys.stdout.buffer.write(pretty_json_bytes(failure, ensure_ascii=True))
        return 1
    sys.stdout.buffer.write(pretty_json_bytes(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Target-free duplicate diagnostic V5 runner.

This module implements the standalone append-only V5 contract, incorporating
the immutable V3 contract and exact V4 serialization correction by explicit
reference and binding the immutable V3 failure incident. Importing the module
is side-effect free. In particular, import never opens Parquet data, fits a
model, creates aliases, or creates the output namespace.

The published preregistration alone authorizes source and synthetic-test
construction only. A real invocation validates the separately sealed
CODE_SEAL/AUTHORIZATION/REVIEW/GO chain and every predata guard before an
Arrow or pandas Parquet API is called.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import stat as stat_module
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol, TypeAlias


SCHEMA_VERSION: Final[int] = 5
REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT_DEFAULT: Final[Path] = (
    REPOSITORY_ROOT
    / "artifacts"
    / "baram2026_ncei_scada_longrun_20260810_v2"
)
AMENDMENT_RELATIVE_PATH: Final[str] = (
    "prereg/target_free_multiseason_amendment_v3.json"
)
AMENDMENT_SIZE_BYTES: Final[int] = 102_361
AMENDMENT_SHA256: Final[str] = (
    "f9d072fc24ab29608efc6483098ada5a8d5e8129e1d1bd478e1d84438a2e9c14"
)
AMENDMENT_CANONICAL_SIZE_BYTES: Final[int] = 84_601
AMENDMENT_CANONICAL_SHA256: Final[str] = (
    "ae5c5d05e63169112799c48a8ede52e8ce29319bc5d9e96927175ccbfd9b91d1"
)
AMENDMENT_V4_RELATIVE_PATH: Final[str] = (
    "prereg/target_free_multiseason_amendment_v4.json"
)
AMENDMENT_V4_SIZE_BYTES: Final[int] = 14_385
AMENDMENT_V4_SHA256: Final[str] = (
    "9f8abbe878bd10e7b91f78a03f44389a4973c8c1cb7da3c94752487957be7980"
)
AMENDMENT_V4_CANONICAL_SIZE_BYTES: Final[int] = 12_501
AMENDMENT_V4_CANONICAL_SHA256: Final[str] = (
    "3ad1f38508b373aac70ee19b4253e91b9104f9527cb77952121757226f35abed"
)
AMENDMENT_V5_RELATIVE_PATH: Final[str] = (
    "prereg/target_free_multiseason_amendment_v5.json"
)
AMENDMENT_V5_SIZE_BYTES: Final[int] = 59_798
AMENDMENT_V5_SHA256: Final[str] = (
    "3e5515f275cd572def713d640aee9d5ecedb982f98d4983ee2d8adcdff0c5d9b"
)
AMENDMENT_V5_CANONICAL_SIZE_BYTES: Final[int] = 48_167
AMENDMENT_V5_CANONICAL_SHA256: Final[str] = (
    "7533dd4ef5e4be293109409ed17730ce10a855dfb8da7f9f0a66dd8e8ef9371f"
)
AMENDMENT_V5_CREATED_UTC: Final[str] = "2026-08-11T18:03:24.805000Z"
V3_FAILURE_INCIDENT_RELATIVE_PATH: Final[str] = (
    "incidents/TARGET_FREE_MULTISEASON_V3_EXECUTION_OR_PUBLICATION_FAILURE_V1.json"
)
V3_FAILURE_INCIDENT_SIZE_BYTES: Final[int] = 44_709
V3_FAILURE_INCIDENT_SHA256: Final[str] = (
    "b5dc9dc117f8686bb7d80f73e082c895c55af832e8b25d33d50661cb19329759"
)
V3_FAILURE_INCIDENT_CANONICAL_SIZE_BYTES: Final[int] = 35_456
V3_FAILURE_INCIDENT_CANONICAL_SHA256: Final[str] = (
    "6d2cc2593ea4375ddb88e6803057595ea34c04eac04be917a07c185eac2bec7a"
)
V5_FAILURE_INCIDENT_RELATIVE_PATH: Final[str] = (
    "incidents/TARGET_FREE_MULTISEASON_V5_EXECUTION_OR_PUBLICATION_FAILURE_V1.json"
)
PARQUET_SERIALIZATION_LITERAL_V3: Final[str] = (
    "PYARROW_22_0_0_PARQUET_VERSION_2_6_ZSTD_LEVEL_3_USE_DICTIONARY_FALSE_"
    "WRITE_STATISTICS_TRUE_DATA_PAGE_VERSION_1_0_ROW_GROUP_SIZE_65536_INDEX_"
    "FALSE_COERCE_TIMESTAMPS_NS_ALLOW_TRUNCATED_TIMESTAMPS_FALSE_USE_"
    "DEPRECATED_INT96_FALSE_STORE_SCHEMA_TRUE_WRITE_PAGE_INDEX_FALSE_FLOAT_"
    "COLUMNS_BINARY64"
)
PARQUET_SERIALIZATION_LITERAL_V4: Final[str] = (
    "PYARROW_22_0_0_PARQUET_VERSION_2_6_ZSTD_LEVEL_3_USE_DICTIONARY_FALSE_"
    "WRITE_STATISTICS_TRUE_DATA_PAGE_VERSION_1_0_ROW_GROUP_SIZE_65536_INDEX_"
    "FALSE_COERCE_TIMESTAMPS_NONE_EFFECTIVE_NATIVE_NS_FOR_VERSION_2_6_ALLOW_"
    "TRUNCATED_TIMESTAMPS_FALSE_USE_DEPRECATED_INT96_FALSE_STORE_SCHEMA_TRUE_"
    "WRITE_PAGE_INDEX_FALSE_FLOAT_COLUMNS_BINARY64"
)
CSV_BOUNDARY_SERIALIZATION_LITERAL_V3: Final[str] = (
    "UPPERCASE_TRUE_OR_FALSE_SEMICOLON_JOINED_IN_FROZEN_FLAG_NAME_ORDER"
)
CSV_BOUNDARY_SERIALIZATION_LITERAL_V4: Final[str] = (
    "CSV_BOOLEAN_CELLS_UPPERCASE_TRUE_OR_FALSE;BOUNDARY_EQUALITY_FLAGS_EMPTY_"
    "IF_NONE_ELSE_SEMICOLON_JOINED_ACTIVE_CONTINUOUS_CUTOFF_NAMES_IN_"
    "COMPONENT_DECISION_TRUTH_TABLE_CONTINUOUS_CUTOFFS_EXACT_ORDER_WITH_NO_"
    "DUPLICATES"
)
PARQUET_WRITE_TABLE_KWARGS_EXACT: Final[Mapping[str, Any]] = {
    "version": "2.6",
    "compression": "zstd",
    "compression_level": 3,
    "use_dictionary": False,
    "write_statistics": True,
    "data_page_version": "1.0",
    "row_group_size": 65_536,
    "use_deprecated_int96_timestamps": False,
    "coerce_timestamps": None,
    "allow_truncated_timestamps": False,
    "store_schema": True,
    "write_page_index": False,
}
PYARROW_WRITE_TABLE_UNLISTED_DEFAULTS_EXACT: Final[Mapping[str, Any]] = {
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

OUTPUT_ROOT_RELATIVE: Final[str] = "modeling/target_free_multiseason_v5"
TRANSACTION_DIRNAME: Final[str] = ".transaction_v5"
V3_FAILED_OUTPUT_ROOT_RELATIVE: Final[str] = (
    "modeling/target_free_multiseason_v3"
)

RESPONSES: Final[tuple[str, ...]] = (
    "HPBL_surface",
    "UGRD_925mb",
    "VGRD_925mb",
    "UGRD_950mb",
    "VGRD_950mb",
    "UGRD_975mb",
    "VGRD_975mb",
    "UGRD_1000mb",
    "VGRD_1000mb",
)
WIND_RESPONSES: Final[tuple[str, ...]] = RESPONSES[1:]
PBL_RESPONSES: Final[tuple[str, ...]] = RESPONSES[:1]

PREDICTORS: Final[tuple[str, ...]] = (
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_80_u",
    "heightAboveGround_80_v",
    "heightAboveGround_100_100u",
    "heightAboveGround_100_100v",
    "heightAboveGround_2_2t",
    "heightAboveGround_2_2d",
    "heightAboveGround_2_2r",
    "heightAboveGround_2_2sh",
    "planetaryBoundaryLayer_0_u",
    "planetaryBoundaryLayer_0_v",
    "planetaryBoundaryLayer_0_VRATE",
    "surface_0_dswrf",
    "surface_0_dlwrf",
    "surface_0_prate",
    "surface_0_tp",
    "surface_0_sp",
    "meanSea_0_prmsl",
    "surface_0_gust",
    "lowCloudLayer_0_lcc",
    "middleCloudLayer_0_mcc",
    "highCloudLayer_0_hcc",
    "atmosphere_0_tcc",
    "isobaricInhPa_850_t",
    "isobaricInhPa_850_u",
    "isobaricInhPa_850_v",
    "isobaricInhPa_850_r",
    "isobaricInhPa_700_t",
    "isobaricInhPa_700_u",
    "isobaricInhPa_700_v",
    "isobaricInhPa_500_gh",
    "isobaricInhPa_500_t",
    "isobaricInhPa_500_u",
    "isobaricInhPa_500_v",
)

SITE_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "valid_time_utc",
    "target_operating_day_kst",
    "forecast_kst_dtm",
    "data_available_kst_dtm",
    "site_id",
    "group",
    "latitude",
    "longitude",
    "capacity_mw",
)
GROUP_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "valid_time_utc",
    "target_operating_day_kst",
    "forecast_kst_dtm",
    "data_available_kst_dtm",
    "group",
    "site_count",
    "capacity_mw",
)
SITE_KEY: Final[tuple[str, str]] = ("valid_time_utc", "site_id")
GROUP_KEY: Final[tuple[str, str]] = ("valid_time_utc", "group")

EXPECTED_SITE_ROWS: Final[int] = 19_584
EXPECTED_GROUP_ROWS: Final[int] = 3_456
EXPECTED_TIMESTAMPS: Final[int] = 1_152
EXPECTED_PRIMARY_UNITS: Final[int] = 72
EXPECTED_AFFINE_UNITS: Final[int] = 1_260
EXPECTED_LEDGER_ROWS: Final[int] = 1_332
EXPECTED_OOF_ROWS: Final[int] = 352_512
EXPECTED_DERIVED_ROWS: Final[int] = 414_720
EXPECTED_DUPLICATE_METRIC_ROWS: Final[int] = 333
EXPECTED_STABILITY_ROWS: Final[int] = 348
EXPECTED_DISTRIBUTION_ROWS: Final[int] = 624
EXPECTED_ENDPOINT_POOLED_ROWS: Final[int] = 3

UNIT_COMPLETED: Final[str] = "COMPLETED"
UNIT_SKIPPED: Final[str] = "SKIPPED_PREMODEL_CLEAR_VETO"
UNIT_STATUSES: Final[frozenset[str]] = frozenset(
    {UNIT_COMPLETED, UNIT_SKIPPED}
)
INVALID_OR_SKIP_REASONS: Final[tuple[str, ...]] = (
    "SKIPPED_PREMODEL_CLEAR_PHYSICAL_WIND_FAMILY_VETO",
    "SKIPPED_PREMODEL_CLEAR_PHYSICAL_PBL_FAMILY_VETO",
    "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3",
    "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_POOLED_SST_LTE_1E_MINUS_12",
    "UNUSABLE_CELL_UNIQUE_COUNT_LT_3",
    "UNUSABLE_CELL_SST_LTE_1E_MINUS_12",
    "UNUSABLE_MONTHLY_IQR_2022_LTE_1E_MINUS_12",
    "UNUSABLE_MONTHLY_IQR_2023_LTE_1E_MINUS_12",
    "UNUSABLE_MONTHLY_IQR_BOTH_YEARS_LTE_1E_MINUS_12",
    "UNUSABLE_PSI_FINAL_BIN_COUNT_LT_2",
    "DEGENERATE_AFFINE_PREDICTOR_SXX_LTE_1E_MINUS_12",
    "DEGENERATE_SPEARMAN_RANK_VARIANCE_LTE_1E_MINUS_12",
    "STRUCTURALLY_NOT_APPLICABLE",
)
INVALID_OR_SKIP_REASON_SET: Final[frozenset[str]] = frozenset(
    INVALID_OR_SKIP_REASONS
)

ENVIRONMENT_CAPS: Final[Mapping[str, str]] = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}


class ContractError(RuntimeError):
    """Base class for a deterministic V5 contract violation."""


class StrictJSONError(ContractError):
    """Strict JSON parsing or serialization failed."""


class IdentityError(ContractError):
    """A path or byte identity did not match the preregistration."""


class PredataAuthorizationError(ContractError):
    """A logical value API was reached before all gates passed."""


class IntegrityError(ContractError):
    """Schema, key, join, provenance, or finite-value integrity failed."""


class GlobalAmbiguity(ContractError):
    """A required comparison is ambiguous under the V3 truth table."""


class OutputPublicationError(ContractError):
    """Append-only output publication failed or would overwrite a path."""


def validate_validity_reason(value: Any, reason: Any, label: str) -> None:
    """Bind valid/null evidence to the amendment's exact13 reason enum."""

    if not isinstance(value, bool):
        raise IntegrityError(f"{label} valid flag is not bool")
    if value:
        if reason is not None:
            raise IntegrityError(f"{label} valid row has an invalid reason")
    elif reason not in INVALID_OR_SKIP_REASON_SET:
        raise IntegrityError(f"{label} invalid reason is outside exact13")


JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)


def _reject_json_constant(token: str) -> None:
    raise StrictJSONError(f"forbidden JSON numeric constant: {token}")


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _walk_finite_json(value: Any, pointer: str = "") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise StrictJSONError(f"non-finite JSON number at {pointer or '/'}")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise StrictJSONError(f"non-string JSON key at {pointer or '/'}")
            escaped = key.replace("~", "~0").replace("/", "~1")
            _walk_finite_json(child, f"{pointer}/{escaped}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_finite_json(child, f"{pointer}/{index}")


def strict_json_loads(payload: str | bytes | bytearray) -> Any:
    """Parse JSON while rejecting duplicate keys and non-finite constants."""

    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except StrictJSONError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise StrictJSONError(str(exc)) from exc
    _walk_finite_json(parsed)
    return parsed


def strict_json_dumps(value: Any, *, pretty: bool) -> bytes:
    """Serialize finite JSON using the inherited deterministic serializer."""

    _walk_finite_json(value)
    try:
        if pretty:
            text = json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            return (text + "\n").encode("utf-8")
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StrictJSONError(str(exc)) from exc


def canonical_amendment_bytes(value: Any) -> bytes:
    """Serialize a parsed amendment with its Python-pinned canonical format."""

    _walk_finite_json(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StrictJSONError(str(exc)) from exc


def pretty_amendment_bytes(value: Any) -> bytes:
    """Serialize an amendment using its ensure-ASCII-false physical format."""

    _walk_finite_json(value)
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        return (text + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StrictJSONError(str(exc)) from exc


@dataclasses.dataclass(frozen=True, slots=True)
class FileIdentity:
    path: str
    size_bytes: int
    sha256: str

    def validate_shape(self) -> None:
        if not self.path or Path(self.path).is_absolute():
            raise IdentityError("bound identity path must be non-empty and relative")
        if self.size_bytes < 0:
            raise IdentityError("bound identity size_bytes must be non-negative")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise IdentityError("bound identity sha256 must be lowercase hex")


V3_FAILED_ALIAS_IDENTITIES: Final[tuple[FileIdentity, ...]] = (
    FileIdentity(
        "modeling/target_free_multiseason_v3/FIELD_CENSUS_LOCK.json",
        4_523,
        "0269f6717b1110fafb6bba052c78824c616c648dbbdfe9f630947012c3f571a1",
    ),
    FileIdentity(
        "modeling/target_free_multiseason_v3/PROVENANCE_LEDGER.parquet",
        3_812_043,
        "11849682fd1c65ca20cfcb3002d3708beccf10ef862bb0f03d054c2c030790d5",
    ),
)


def _sha256_and_size(path: Path, chunk_size: int = 1 << 20) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _linklike(path: Path) -> bool:
    """Return true for symbolic links, junctions, or other reparse points."""

    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return False
        reparse_flag = int(
            getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
        return bool(
            int(getattr(metadata, "st_file_attributes", 0)) & reparse_flag
        )
    except OSError:
        # A reparse-point query that cannot be completed is not safe to follow.
        return True


def _absolute_path_components(path: Path) -> tuple[Path, ...]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.is_absolute() or not absolute.anchor:
        raise IdentityError(f"path is not absolute: {path}")
    current = Path(absolute.anchor)
    result: list[Path] = [current]
    for part in absolute.parts[1:]:
        current /= part
        result.append(current)
    return tuple(result)


def _assert_no_linklike_absolute(path: Path, label: str) -> None:
    for component in _absolute_path_components(path):
        if _linklike(component):
            raise IdentityError(
                f"{label} contains a symlink or junction or other reparse point"
            )


def _normcase_lexical(path: str | os.PathLike[str]) -> str:
    """Case-normalize a lexical path without making it absolute/resolved."""

    return os.path.normcase(os.fspath(path))


def _validate_canonical_absolute_path(
    supplied: str | os.PathLike[str],
    expected: Path,
    *,
    label: str,
) -> Path:
    """Validate an exact lexical CLI path before following any component."""

    if not isinstance(supplied, (str, os.PathLike)):
        raise PredataAuthorizationError(f"{label} CLI path type differs")
    lexical_text = os.fspath(supplied)
    if not isinstance(lexical_text, str):
        raise PredataAuthorizationError(f"{label} CLI path is not text")
    lexical = Path(lexical_text)
    if not lexical.is_absolute():
        raise PredataAuthorizationError(f"{label} CLI path must be absolute")
    expected_text = os.fspath(expected)
    if _normcase_lexical(lexical_text) != _normcase_lexical(expected_text):
        raise PredataAuthorizationError(
            f"{label} CLI path is not the frozen canonical lexical path"
        )
    try:
        _assert_no_linklike_absolute(lexical, f"{label} CLI path")
        resolved = lexical.resolve(strict=True)
    except (IdentityError, OSError) as exc:
        raise PredataAuthorizationError(
            f"{label} CLI path is link-like, absent, or unreadable"
        ) from exc
    if _normcase_lexical(resolved) != _normcase_lexical(expected):
        raise PredataAuthorizationError(
            f"{label} CLI path resolves away from the frozen canonical path"
        )
    return resolved


def validate_production_cli_paths(
    root_text: str,
    authorization_text: str,
    independent_go_text: str,
    *,
    expected_root: Path = ARTIFACT_ROOT_DEFAULT,
) -> tuple[Path, Path, Path]:
    """Bind raw production CLI path lexemes to the frozen absolute paths."""

    canonical_root = _validate_canonical_absolute_path(
        root_text, expected_root, label="artifact root"
    )
    canonical_authorization = _validate_canonical_absolute_path(
        authorization_text,
        canonical_root / EXECUTION_AUTHORIZATION_RELATIVE_PATH,
        label="authorization",
    )
    canonical_go = _validate_canonical_absolute_path(
        independent_go_text,
        canonical_root / INDEPENDENT_GO_RELATIVE_PATH,
        label="independent GO",
    )
    return canonical_root, canonical_authorization, canonical_go


def _assert_no_linklike_relative(root: Path, relative_path: str, label: str) -> None:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise IdentityError(f"{label} is not a contained relative path")
    unresolved_root = Path(os.path.abspath(os.fspath(root)))
    _assert_no_linklike_absolute(unresolved_root, f"{label} root")
    _assert_no_linklike_absolute(unresolved_root / relative, label)


def resolve_relative_identity(root: Path, identity: FileIdentity) -> Path:
    """Resolve a bound relative path without permitting root escape."""

    identity.validate_shape()
    _assert_no_linklike_relative(root, identity.path, "bound identity path")
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / Path(identity.path)).resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise IdentityError(f"identity escapes artifact root: {identity.path}") from exc
    if not candidate.is_file():
        raise IdentityError(f"identity is not a regular file: {identity.path}")
    return candidate


def revalidate_file_identity(root: Path, identity: FileIdentity) -> Path:
    path = resolve_relative_identity(root, identity)
    size, digest = _sha256_and_size(path)
    if size != identity.size_bytes or digest != identity.sha256:
        raise IdentityError(
            f"identity mismatch for {identity.path}: "
            f"observed size={size} sha256={digest}"
        )
    return path


def validate_amendment(root: Path) -> Mapping[str, Any]:
    identity = FileIdentity(
        AMENDMENT_RELATIVE_PATH,
        AMENDMENT_SIZE_BYTES,
        AMENDMENT_SHA256,
    )
    path = revalidate_file_identity(root, identity)
    raw = path.read_bytes()
    amendment = strict_json_loads(raw)
    if not isinstance(amendment, dict):
        raise StrictJSONError("amendment top-level value must be an object")
    canonical = canonical_amendment_bytes(amendment)
    if len(canonical) != AMENDMENT_CANONICAL_SIZE_BYTES:
        raise IdentityError("amendment canonical byte count mismatch")
    if hashlib.sha256(canonical).hexdigest() != AMENDMENT_CANONICAL_SHA256:
        raise IdentityError("amendment canonical sha256 mismatch")
    return amendment


AMENDMENT_V4_TOP_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "artifact_type",
    "status",
    "created_utc",
    "canonical_path",
    "bound_v3_amendment",
    "correction_scope",
    "pyarrow_22_0_0_evidence",
    "corrected_parquet_serialization",
    "corrected_csv_boundary_serialization",
    "affected_output_schemas_exact",
    "preserved_v3_semantics",
    "authority_scope",
    "required_future_control_bindings",
    "final_lock_manifest_binding_supersession",
    "required_tests",
    "prohibitions",
    "strict_schema",
)
AMENDMENT_V5_TOP_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "artifact_type",
    "status",
    "created_utc",
    "canonical_path",
    "bound_v3_amendment",
    "bound_v4_amendment",
    "bound_v3_failure_incident",
    "inherited_effective_contract",
    "correction_scope",
    "bound_parquet_footer_schemas_exact",
    "predata_footer_gate_and_execution_order",
    "v3_failure_root_preservation",
    "code_and_test_paths_exact",
    "control_contract",
    "output_contract",
    "execution_budget",
    "runtime_and_serialization_preservation",
    "final_lock_manifest_binding",
    "required_tests",
    "release_workflow",
    "authority_scope",
    "prohibitions",
    "strict_schema",
)
V4_SUPERSEDED_POINTERS: Final[tuple[str, ...]] = (
    "/output_contract/output_serialization_exact/parquet",
    "/output_contract/output_serialization_exact/csv_boundary_flags",
)
V4_REPLACEMENT_LITERALS: Final[Mapping[str, str]] = {
    V4_SUPERSEDED_POINTERS[0]: PARQUET_SERIALIZATION_LITERAL_V4,
    V4_SUPERSEDED_POINTERS[1]: CSV_BOUNDARY_SERIALIZATION_LITERAL_V4,
}
V3_SUPERSEDED_LITERALS: Final[Mapping[str, str]] = {
    V4_SUPERSEDED_POINTERS[0]: PARQUET_SERIALIZATION_LITERAL_V3,
    V4_SUPERSEDED_POINTERS[1]: CSV_BOUNDARY_SERIALIZATION_LITERAL_V3,
}
CODE_CONFIG_RUNTIME_IDENTITY_KEYS: Final[tuple[str, ...]] = (
    "amendment",
    "amendment_v4",
    "amendment_v5",
    "code_identities",
    "runtime_identity",
)


def _json_pointer_get(value: Mapping[str, Any], pointer: str) -> Any:
    current: Any = value
    if not pointer.startswith("/"):
        raise IdentityError("JSON pointer must be absolute")
    for raw_part in pointer.split("/")[1:]:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise IdentityError(f"JSON pointer is absent: {pointer}")
        current = current[part]
    return current


def _expected_v4_authority_scope() -> dict[str, Any]:
    return {
        "any_other_file_write_authorized": False,
        "arrays_2024_or_2025_read_authorized": False,
        "authorized_code_and_test_paths": [
            "scripts/run_noaa_gfs_target_free_duplicate_v3.py",
            "tests/test_noaa_gfs_target_free_duplicate_v3.py",
            "scripts/audit_noaa_gfs_target_free_duplicate_v3.py",
            "tests/test_audit_noaa_gfs_target_free_duplicate_v3.py",
            "scripts/seal_noaa_gfs_target_free_duplicate_v3.py",
            "tests/test_seal_noaa_gfs_target_free_duplicate_v3.py",
        ],
        "code_and_test_implementation_authorized": True,
        "control_publication_authorized": False,
        "decoded_or_predictor_parquet_value_read_authorized": False,
        "documentary_correction": True,
        "execution_authorized": False,
        "execution_output_publication_authorized": False,
        "gpu_authorized": False,
        "label_or_scada_read_authorized": False,
        "model_or_analytic_fit_authorized": False,
        "network_authorized": False,
        "public_feedback_or_prediction_read_authorized": False,
        "submission_csv_authorized": False,
        "upstream_alias_write_authorized": False,
    }


def validate_v3_v4_overlay(
    amendment: Mapping[str, Any], amendment_v4: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Apply only the two V4 documentary pointer replacements in memory."""

    scope = amendment_v4.get("correction_scope")
    if not isinstance(scope, dict) or set(scope) != {
        "all_other_v3_parsed_semantic_pointers_unchanged",
        "correction_count_exact",
        "replacement_literals_exact",
        "scope_expansion_forbidden",
        "superseded_pointers_exact_order",
        "v3_literals_exact",
    }:
        raise IdentityError("V4 correction scope schema differs")
    if (
        scope["correction_count_exact"] != 2
        or scope["superseded_pointers_exact_order"]
        != list(V4_SUPERSEDED_POINTERS)
        or scope["v3_literals_exact"] != dict(V3_SUPERSEDED_LITERALS)
        or scope["replacement_literals_exact"]
        != dict(V4_REPLACEMENT_LITERALS)
        or scope["all_other_v3_parsed_semantic_pointers_unchanged"] is not True
        or scope["scope_expansion_forbidden"] is not True
    ):
        raise IdentityError("V4 correction scope values differ")
    for pointer in V4_SUPERSEDED_POINTERS:
        if _json_pointer_get(amendment, pointer) != V3_SUPERSEDED_LITERALS[pointer]:
            raise IdentityError(f"V3 superseded literal differs: {pointer}")
    serialization = dict(
        amendment["output_contract"]["output_serialization_exact"]
    )
    serialization["parquet"] = PARQUET_SERIALIZATION_LITERAL_V4
    serialization["csv_boundary_flags"] = CSV_BOUNDARY_SERIALIZATION_LITERAL_V4
    changed = {
        key
        for key, value in serialization.items()
        if value
        != amendment["output_contract"]["output_serialization_exact"][key]
    }
    if changed != {"parquet", "csv_boundary_flags"}:
        raise IdentityError("V4 overlay changed a non-authorized pointer")
    return serialization


def validate_amendment_v4(
    root: Path, amendment: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    identity = FileIdentity(
        AMENDMENT_V4_RELATIVE_PATH,
        AMENDMENT_V4_SIZE_BYTES,
        AMENDMENT_V4_SHA256,
    )
    path = revalidate_file_identity(root, identity)
    raw = path.read_bytes()
    amendment_v4 = strict_json_loads(raw)
    if not isinstance(amendment_v4, dict):
        raise StrictJSONError("V4 amendment top-level value must be an object")
    canonical = canonical_amendment_bytes(amendment_v4)
    if (
        len(canonical) != AMENDMENT_V4_CANONICAL_SIZE_BYTES
        or hashlib.sha256(canonical).hexdigest()
        != AMENDMENT_V4_CANONICAL_SHA256
    ):
        raise IdentityError("V4 amendment canonical identity mismatch")
    if (
        set(amendment_v4) != set(AMENDMENT_V4_TOP_KEYS)
        or len(amendment_v4) != len(AMENDMENT_V4_TOP_KEYS)
        or amendment_v4["schema_version"] != 4
        or amendment_v4["artifact_type"]
        != "TARGET_FREE_MULTISEASON_PREREGISTRATION_APPEND_ONLY_DOCUMENTARY_CORRECTION_V4"
        or amendment_v4["status"]
        != "PASS_DOCUMENTARY_V4_PARQUET_API_AND_CSV_BOUNDARY_SERIALIZATION_CORRECTION_ONLY_NO_EXECUTION_AUTHORITY"
        or amendment_v4["canonical_path"] != AMENDMENT_V4_RELATIVE_PATH
    ):
        raise IdentityError("V4 amendment envelope differs")
    if amendment is None:
        amendment = validate_amendment(root)
    if amendment_v4["bound_v3_amendment"] != _expected_amendment_identity():
        raise IdentityError("V4 bound V3 amendment identity differs")
    parquet = amendment_v4["corrected_parquet_serialization"]
    csv_boundary = amendment_v4["corrected_csv_boundary_serialization"]
    if (
        parquet["corrected_v3_literal"] != PARQUET_SERIALIZATION_LITERAL_V4
        or parquet["explicit_write_table_kwargs_exact"]
        != dict(PARQUET_WRITE_TABLE_KWARGS_EXACT)
        or csv_boundary["corrected_v3_literal"]
        != CSV_BOUNDARY_SERIALIZATION_LITERAL_V4
        or csv_boundary["csv_boolean_cells_exact_values"]
        != ["TRUE", "FALSE"]
        or amendment_v4["authority_scope"] != _expected_v4_authority_scope()
    ):
        raise IdentityError("V4 corrected serialization or authority differs")
    validate_v3_v4_overlay(amendment, amendment_v4)
    return amendment_v4


V3_FAILURE_INCIDENT_TOP_KEYS: Final[tuple[str, ...]] = (
    "amendments",
    "artifact_type",
    "attempt_id",
    "authority_scope",
    "canonical_path",
    "code_identities",
    "control_identities",
    "created_utc",
    "failure_boundary",
    "footer_schema_evidence",
    "forbidden_access_attestation",
    "postfailure_state",
    "prohibitions",
    "publisher",
    "required_v5_supersession",
    "runner_execution_capture",
    "schema_version",
    "status",
    "strict_schema",
    "v3_disposition",
)
V5_TOP_OBJECT_KEY_COUNTS: Final[Mapping[str, int]] = {
    "authority_scope": 22,
    "bound_parquet_footer_schemas_exact": 8,
    "bound_v3_amendment": 5,
    "bound_v3_failure_incident": 5,
    "bound_v4_amendment": 5,
    "code_and_test_paths_exact": 6,
    "control_contract": 26,
    "correction_scope": 12,
    "execution_budget": 14,
    "final_lock_manifest_binding": 9,
    "inherited_effective_contract": 15,
    "output_contract": 19,
    "predata_footer_gate_and_execution_order": 12,
    "prohibitions": 36,
    "release_workflow": 12,
    "required_tests": 4,
    "runtime_and_serialization_preservation": 10,
    "strict_schema": 11,
    "v3_failure_root_preservation": 12,
}


def validate_v3_failure_incident(root: Path) -> Mapping[str, Any]:
    """Bind the immutable V3 failed-attempt incident without inferring UTC."""

    identity = FileIdentity(
        V3_FAILURE_INCIDENT_RELATIVE_PATH,
        V3_FAILURE_INCIDENT_SIZE_BYTES,
        V3_FAILURE_INCIDENT_SHA256,
    )
    path = revalidate_file_identity(root, identity)
    raw = path.read_bytes()
    incident = strict_json_loads(raw)
    if not isinstance(incident, dict):
        raise StrictJSONError("V3 failure incident must be an object")
    canonical = canonical_amendment_bytes(incident)
    if (
        len(canonical) != V3_FAILURE_INCIDENT_CANONICAL_SIZE_BYTES
        or hashlib.sha256(canonical).hexdigest()
        != V3_FAILURE_INCIDENT_CANONICAL_SHA256
        or strict_json_dumps(incident, pretty=True) != raw
    ):
        raise IdentityError("V3 failure incident serialization identity differs")
    if (
        set(incident) != set(V3_FAILURE_INCIDENT_TOP_KEYS)
        or len(incident) != 20
        or type(incident.get("schema_version")) is not int
        or incident["schema_version"] != 1
        or incident.get("artifact_type")
        != "TARGET_FREE_DUPLICATE_V3_BOUND_INPUT_FOOTER_SCHEMA_FALSE_REJECT_INCIDENT_V1"
        or incident.get("status")
        != "SEALED_V3_BOUND_INPUT_FOOTER_SCHEMA_FALSE_REJECT_AFTER_ALIAS2_NO_VALUE_READ_NO_FIT_V5_REQUIRED"
        or incident.get("canonical_path") != V3_FAILURE_INCIDENT_RELATIVE_PATH
    ):
        raise IdentityError("V3 failure incident envelope differs")
    strict = incident.get("strict_schema")
    if (
        not isinstance(strict, dict)
        or strict.get("top_level_keys_exact")
        != list(V3_FAILURE_INCIDENT_TOP_KEYS)
        or strict.get("top_level_key_count") != 20
        or strict.get("type_exact_validation_required") is not True
    ):
        raise IdentityError("V3 failure incident strict schema differs")
    capture = incident.get("runner_execution_capture")
    boundary = incident.get("failure_boundary")
    if not isinstance(capture, dict) or not isinstance(boundary, dict):
        raise IdentityError("V3 failure incident capture boundary is absent")
    embedded = capture.get("capture_record_embedded_exact")
    if (
        capture.get("start_end_utc_instrumented") is not False
        or capture.get("start_utc") is not None
        or capture.get("end_utc") is not None
        or not isinstance(embedded, dict)
        or embedded.get("start_end_utc_instrumented") is not False
        or boundary.get("any_bound_parquet_logical_value_read") is not False
        or boundary.get("any_fit_reached") is not False
        or boundary.get("arrow_handoff_authorized") is not True
        or boundary.get("guarded_read_table_reached") is not False
        or boundary.get("run_prepared_execution_reached") is not False
    ):
        raise IdentityError("V3 failure capture/no-value/no-fit closure differs")
    return incident


def validate_amendment_v5(
    root: Path,
    amendment: Mapping[str, Any] | None = None,
    amendment_v4: Mapping[str, Any] | None = None,
    incident: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate the standalone V5 amendment and its immutable lineage."""

    identity = FileIdentity(
        AMENDMENT_V5_RELATIVE_PATH,
        AMENDMENT_V5_SIZE_BYTES,
        AMENDMENT_V5_SHA256,
    )
    path = revalidate_file_identity(root, identity)
    raw = path.read_bytes()
    amendment_v5 = strict_json_loads(raw)
    if not isinstance(amendment_v5, dict):
        raise StrictJSONError("V5 amendment top-level value must be an object")
    canonical = canonical_amendment_bytes(amendment_v5)
    if (
        len(canonical) != AMENDMENT_V5_CANONICAL_SIZE_BYTES
        or hashlib.sha256(canonical).hexdigest()
        != AMENDMENT_V5_CANONICAL_SHA256
        or pretty_amendment_bytes(amendment_v5) != raw
    ):
        raise IdentityError("V5 amendment serialization identity differs")
    if (
        set(amendment_v5) != set(AMENDMENT_V5_TOP_KEYS)
        or len(amendment_v5) != len(AMENDMENT_V5_TOP_KEYS)
        or type(amendment_v5.get("schema_version")) is not int
        or amendment_v5["schema_version"] != 5
        or amendment_v5.get("artifact_type")
        != "TARGET_FREE_MULTISEASON_PREREGISTRATION_APPEND_ONLY_EXECUTION_SUPERSESSION_V5"
        or amendment_v5.get("status")
        != "FROZEN_FOR_V5_CODE_AND_TEST_IMPLEMENTATION_ONLY_PENDING_SEPARATE_CODE_SEAL_REVIEW_AND_GO_BEFORE_FOOTER_GATE_VALUE_READ_OR_FIT"
        or amendment_v5.get("created_utc") != AMENDMENT_V5_CREATED_UTC
        or amendment_v5.get("canonical_path") != AMENDMENT_V5_RELATIVE_PATH
    ):
        raise IdentityError("V5 amendment envelope differs")
    strict = amendment_v5.get("strict_schema")
    if (
        not isinstance(strict, dict)
        or strict.get("top_level_exact_keys") != list(AMENDMENT_V5_TOP_KEYS)
        or strict.get("top_level_exact_key_count") != len(AMENDMENT_V5_TOP_KEYS)
        or strict.get("top_level_object_key_counts")
        != dict(V5_TOP_OBJECT_KEY_COUNTS)
    ):
        raise IdentityError("V5 strict-schema inventory differs")
    for key, count in V5_TOP_OBJECT_KEY_COUNTS.items():
        nested = amendment_v5.get(key)
        if not isinstance(nested, dict) or len(nested) != count:
            raise IdentityError(f"V5 nested key count differs: {key}")
    if amendment is None:
        amendment = validate_amendment(root)
    if amendment_v4 is None:
        amendment_v4 = validate_amendment_v4(root, amendment)
    if incident is None:
        incident = validate_v3_failure_incident(root)
    if (
        amendment_v5["bound_v3_amendment"] != _expected_amendment_identity()
        or amendment_v5["bound_v4_amendment"]
        != _expected_amendment_v4_identity()
        or amendment_v5["bound_v3_failure_incident"]
        != _expected_v3_failure_incident_identity()
        or incident["amendments"]
        != {
            "amendment": _expected_amendment_identity(),
            "amendment_v4": _expected_amendment_v4_identity(),
        }
    ):
        raise IdentityError("V5 immutable lineage crosslink differs")
    return amendment_v5


@dataclasses.dataclass(frozen=True, slots=True)
class Fold:
    ordinal: int
    name: str
    start_operating_day: dt.date
    end_operating_day: dt.date


FOLDS: Final[tuple[Fold, ...]] = (
    Fold(1, "2022_H1", dt.date(2022, 1, 1), dt.date(2022, 6, 30)),
    Fold(2, "2022_H2", dt.date(2022, 7, 1), dt.date(2022, 12, 31)),
    Fold(3, "2023_H1", dt.date(2023, 1, 1), dt.date(2023, 6, 30)),
    Fold(4, "2023_H2", dt.date(2023, 7, 1), dt.date(2023, 12, 31)),
)


def fold_ordinal_for_operating_day(value: dt.date) -> int:
    matches = [
        fold.ordinal
        for fold in FOLDS
        if fold.start_operating_day <= value <= fold.end_operating_day
    ]
    if len(matches) != 1:
        raise IntegrityError(
            f"operating day is outside or ambiguously inside folds: {value}"
        )
    return matches[0]


@dataclasses.dataclass(frozen=True, slots=True)
class PlannedUnit:
    planned_unit_slot_id: str
    decision_fit_ordinal: int
    unit_kind: str
    response: str
    fold: str
    estimator_unit: str | None
    predictor: str | None


def planned_fit_units() -> tuple[PlannedUnit, ...]:
    result: list[PlannedUnit] = []
    ordinal = 0
    estimators = ("RIDGE_PIPELINE", "EXTRA_TREES")
    for response_ordinal, response in enumerate(RESPONSES, start=1):
        for fold in FOLDS:
            for estimator_ordinal, estimator in enumerate(estimators, start=1):
                ordinal += 1
                result.append(
                    PlannedUnit(
                        planned_unit_slot_id=(
                            f"PMU__{response_ordinal:02d}__"
                            f"{fold.ordinal:01d}__{estimator_ordinal:01d}"
                        ),
                        decision_fit_ordinal=ordinal,
                        unit_kind="PRIMARY_MODEL",
                        response=response,
                        fold=fold.name,
                        estimator_unit=estimator,
                        predictor=None,
                    )
                )
    if ordinal != EXPECTED_PRIMARY_UNITS:
        raise AssertionError("primary unit construction drift")
    affine_ordinal = 0
    for response_ordinal, response in enumerate(RESPONSES, start=1):
        for predictor_ordinal, predictor in enumerate(PREDICTORS, start=1):
            for fold in FOLDS:
                ordinal += 1
                affine_ordinal += 1
                result.append(
                    PlannedUnit(
                        planned_unit_slot_id=(
                            f"AAU__{response_ordinal:02d}__"
                            f"{predictor_ordinal:02d}__{fold.ordinal:01d}"
                        ),
                        decision_fit_ordinal=ordinal,
                        unit_kind="ANALYTIC_AFFINE",
                        response=response,
                        fold=fold.name,
                        estimator_unit=None,
                        predictor=predictor,
                    )
                )
    if affine_ordinal != EXPECTED_AFFINE_UNITS:
        raise AssertionError("affine unit construction drift")
    if len(result) != EXPECTED_LEDGER_ROWS:
        raise AssertionError("fit ledger construction drift")
    slot_ids = [item.planned_unit_slot_id for item in result]
    if len(set(slot_ids)) != len(slot_ids):
        raise AssertionError("duplicate planned unit slot id")
    return tuple(result)


PRIMARY_LEDGER_KEYS: Final[tuple[str, ...]] = (
    "planned_unit_slot_id",
    "decision_fit_ordinal",
    "unit_kind",
    "response",
    "fold",
    "estimator_unit",
    "training_rows",
    "training_columns",
    "heldout_rows",
    "input_dtype",
    "response_dtype",
    "unit_status",
    "skip_reason",
    "pipeline_fit_calls",
    "standard_scaler_fit_calls",
    "ridge_fit_calls",
    "extra_trees_fit_calls",
    "predict_calls",
    "random_state",
    "fit_completed",
)
AFFINE_LEDGER_KEYS: Final[tuple[str, ...]] = (
    "planned_unit_slot_id",
    "decision_fit_ordinal",
    "analytic_affine_ordinal",
    "unit_kind",
    "response",
    "predictor",
    "fold",
    "training_rows",
    "heldout_rows",
    "unit_status",
    "skip_reason",
    "sxx",
    "sxx_float_hex",
    "constant_predictor_branch",
    "slope",
    "slope_float_hex",
    "intercept",
    "intercept_float_hex",
    "fit_completed",
)


def _skipped_primary_record(unit: PlannedUnit, reason: str) -> dict[str, Any]:
    if unit.unit_kind != "PRIMARY_MODEL" or unit.estimator_unit is None:
        raise IntegrityError("primary skipped record received a non-primary unit")
    return {
        "planned_unit_slot_id": unit.planned_unit_slot_id,
        "decision_fit_ordinal": unit.decision_fit_ordinal,
        "unit_kind": unit.unit_kind,
        "response": unit.response,
        "fold": unit.fold,
        "estimator_unit": unit.estimator_unit,
        "training_rows": 14_688,
        "training_columns": 35,
        "heldout_rows": 4_896,
        "input_dtype": (
            "float64_C_CONTIGUOUS"
            if unit.estimator_unit == "RIDGE_PIPELINE"
            else "float32_C_CONTIGUOUS"
        ),
        "response_dtype": "float64",
        "unit_status": UNIT_SKIPPED,
        "skip_reason": reason,
        "pipeline_fit_calls": 0,
        "standard_scaler_fit_calls": 0,
        "ridge_fit_calls": 0,
        "extra_trees_fit_calls": 0,
        "predict_calls": 0,
        "random_state": (
            260810 if unit.estimator_unit == "EXTRA_TREES" else None
        ),
        "fit_completed": False,
    }


def _skipped_affine_record(
    unit: PlannedUnit, reason: str, analytic_affine_ordinal: int
) -> dict[str, Any]:
    if unit.unit_kind != "ANALYTIC_AFFINE" or unit.predictor is None:
        raise IntegrityError("affine skipped record received a non-affine unit")
    return {
        "planned_unit_slot_id": unit.planned_unit_slot_id,
        "decision_fit_ordinal": unit.decision_fit_ordinal,
        "analytic_affine_ordinal": analytic_affine_ordinal,
        "unit_kind": unit.unit_kind,
        "response": unit.response,
        "predictor": unit.predictor,
        "fold": unit.fold,
        "training_rows": 14_688,
        "heldout_rows": 4_896,
        "unit_status": UNIT_SKIPPED,
        "skip_reason": reason,
        "sxx": None,
        "sxx_float_hex": None,
        "constant_predictor_branch": None,
        "slope": None,
        "slope_float_hex": None,
        "intercept": None,
        "intercept_float_hex": None,
        "fit_completed": False,
    }


def skipped_fit_ledger(
    response_skip_reasons: Mapping[str, str],
    *,
    completed_unit_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Materialize all 1,332 fixed slots with completed or skipped status."""

    completed = dict(completed_unit_records or {})
    unknown_responses = set(response_skip_reasons).difference(RESPONSES)
    if unknown_responses:
        raise IntegrityError(
            f"unknown skipped responses: {sorted(unknown_responses)!r}"
        )
    records: list[dict[str, Any]] = []
    affine_ordinal = 0
    for unit in planned_fit_units():
        if unit.unit_kind == "ANALYTIC_AFFINE":
            affine_ordinal += 1
        supplied = completed.pop(unit.planned_unit_slot_id, None)
        reason = response_skip_reasons.get(unit.response)
        if supplied is not None and reason is not None:
            raise IntegrityError("a fit unit is both completed and skipped")
        if supplied is not None:
            record = dict(supplied)
            expected_keys = (
                PRIMARY_LEDGER_KEYS
                if unit.unit_kind == "PRIMARY_MODEL"
                else AFFINE_LEDGER_KEYS
            )
            if tuple(record) != expected_keys:
                raise IntegrityError(
                    f"completed ledger schema differs for {unit.planned_unit_slot_id}"
                )
            if record["planned_unit_slot_id"] != unit.planned_unit_slot_id:
                raise IntegrityError("completed ledger slot identity differs")
            if record["unit_status"] != UNIT_COMPLETED or not record["fit_completed"]:
                raise IntegrityError("completed ledger status differs")
        elif reason is not None:
            if unit.unit_kind == "PRIMARY_MODEL":
                record = _skipped_primary_record(unit, reason)
            else:
                record = _skipped_affine_record(unit, reason, affine_ordinal)
        else:
            raise IntegrityError(
                f"neither completed nor skipped: {unit.planned_unit_slot_id}"
            )
        records.append(record)
    if completed:
        raise IntegrityError(
            f"unknown completed fit unit slots: {sorted(completed)!r}"
        )
    if len(records) != EXPECTED_LEDGER_ROWS:
        raise AssertionError("fit ledger row count drift")
    return tuple(records)


def planned_oof_slot_id(
    response_ordinal: int, estimator_ordinal: int, canonical_site_row_ordinal: int
) -> str:
    if not 1 <= response_ordinal <= 9:
        raise IntegrityError("OOF response ordinal is outside 1..9")
    if estimator_ordinal not in (1, 2):
        raise IntegrityError("OOF estimator ordinal is outside 1..2")
    if not 1 <= canonical_site_row_ordinal <= EXPECTED_SITE_ROWS:
        raise IntegrityError("OOF site row ordinal is outside 1..19584")
    return (
        f"OOF__{response_ordinal:02d}__{estimator_ordinal:01d}__"
        f"{canonical_site_row_ordinal:05d}"
    )


def planned_derived_slot_id(
    diagnostic_ordinal: int, scope: str, scope_row_ordinal: int
) -> str:
    if not 1 <= diagnostic_ordinal <= len(DERIVED_DIAGNOSTICS):
        raise IntegrityError("derived diagnostic ordinal is outside 1..18")
    if scope == "SITE":
        maximum = EXPECTED_SITE_ROWS
        marker = "S"
    elif scope == "GROUP":
        maximum = EXPECTED_GROUP_ROWS
        marker = "G"
    else:
        raise IntegrityError("derived scope must be SITE or GROUP")
    if not 1 <= scope_row_ordinal <= maximum:
        raise IntegrityError("derived scope row ordinal is outside frozen range")
    return f"DWD__{diagnostic_ordinal:02d}__{marker}__{scope_row_ordinal:05d}"


def premodel_response_skip_reason(actual: Any) -> str | None:
    """Return only coherent constant-response skips; non-finite is integrity."""

    np = _np()
    y = _float64_vector(actual, "premodel response")
    unique = int(np.unique(y).size)
    if unique < 3:
        return "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3"
    mean = np.float64(np.sum(y, dtype=np.float64) / np.float64(y.size))
    centered = np.ascontiguousarray(y - mean, dtype=np.float64)
    sst = np.float64(np.sum(centered * centered, dtype=np.float64))
    if bool(sst <= np.float64(SST_GUARD)):
        return (
            "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_"
            "POOLED_SST_LTE_1E_MINUS_12"
        )
    return None


@dataclasses.dataclass(frozen=True, slots=True)
class ControlValidation:
    """Opaque result supplied by the separately sealed control validator.

    Exact future control schemas and statuses are intentionally not duplicated
    here. The code seal freezes one validator implementation, and this runner
    consumes only its conservative pass/fail interface.
    """

    amendment_pass: bool
    code_seal_pass: bool
    execution_authorization_pass: bool
    independent_review_pass: bool
    independent_go_pass: bool
    runtime_identity_pass: bool
    bound_input_identity_and_schema_pass: bool
    output_namespace_absent_pass: bool
    aliases_pass: bool


IDENTITY_KEYS: Final[tuple[str, ...]] = ("path", "size_bytes", "sha256")
AMENDMENT_IDENTITY_KEYS: Final[tuple[str, ...]] = (
    "path",
    "size_bytes",
    "sha256",
    "canonical_size_bytes",
    "canonical_sha256",
)
CODE_IDENTITY_KEYS: Final[tuple[str, ...]] = (
    "runner",
    "runner_test",
    "auditor",
    "auditor_test",
    "sealer",
    "sealer_test",
)
CODE_IDENTITY_FILES: Final[Mapping[str, str]] = {
    "runner": "scripts/run_noaa_gfs_target_free_duplicate_v5.py",
    "runner_test": "tests/test_noaa_gfs_target_free_duplicate_v5.py",
    "auditor": "scripts/audit_noaa_gfs_target_free_duplicate_v5.py",
    "auditor_test": "tests/test_audit_noaa_gfs_target_free_duplicate_v5.py",
    "sealer": "scripts/seal_noaa_gfs_target_free_duplicate_v5.py",
    "sealer_test": "tests/test_seal_noaa_gfs_target_free_duplicate_v5.py",
}
CODE_SEAL_RELATIVE_PATH: Final[str] = (
    "prereg/target_free_duplicate_code_seal_v5.json"
)
EXECUTION_AUTHORIZATION_RELATIVE_PATH: Final[str] = (
    "prereg/target_free_duplicate_execution_authorization_v5.json"
)
INDEPENDENT_REVIEW_RELATIVE_PATH: Final[str] = (
    "independent_redteam/TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_REVIEW_V5.json"
)
INDEPENDENT_GO_RELATIVE_PATH: Final[str] = (
    "independent_redteam/TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_GO_V5.json"
)
POSTRUN_PASS_RELATIVE_PATH: Final[str] = (
    "independent_redteam/TRACK_A_TARGET_FREE_DUPLICATE_POSTRUN_PASS_V5.json"
)

CONTROL_TOP_KEYS: Final[Mapping[str, tuple[str, ...]]] = {
    "CODE_SEAL": (
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "amendment",
        "amendment_v4",
        "amendment_v5",
        "code_identities",
        "test_evidence",
        "output_namespace",
        "authority_scope",
        "required_next_controls",
        "publisher",
    ),
    "AUTHORIZATION": (
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "attempt_id",
        "amendment",
        "amendment_v4",
        "amendment_v5",
        "code_seal",
        "code_identities",
        "output_namespace",
        "execution_budget",
        "required_commands",
        "required_review",
        "required_go",
        "authority_scope",
        "publisher",
    ),
    "REVIEW": (
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "attempt_id",
        "amendment",
        "amendment_v4",
        "amendment_v5",
        "code_seal",
        "authorization",
        "code_identities",
        "output_namespace",
        "independent_checks",
        "authority_scope",
        "verdict",
        "publisher",
    ),
    "GO": (
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "attempt_id",
        "amendment",
        "amendment_v4",
        "amendment_v5",
        "code_seal",
        "authorization",
        "independent_review",
        "code_identities",
        "output_namespace",
        "required_commands",
        "authority_scope",
        "single_attempt",
        "publisher",
    ),
    "POSTRUN": (
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "attempt_id",
        "amendment",
        "amendment_v4",
        "amendment_v5",
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
    ),
}
CONTROL_LITERALS: Final[Mapping[str, Mapping[str, str]]] = {
    "CODE_SEAL": {
        "artifact_type": "TARGET_FREE_DUPLICATE_CODE_SEAL_V5",
        "status": "SEALED_V5_CODE_AND_TESTS_NO_EXECUTION_AUTHORITY",
        "publisher": "V5_SEALER_CODE_SEAL_MODE_CREATE_IF_ABSENT_ONLY",
    },
    "AUTHORIZATION": {
        "artifact_type": "TARGET_FREE_DUPLICATE_EXECUTION_AUTHORIZATION_V5",
        "status": (
            "AUTHORIZED_V5_SINGLE_ATTEMPT_PENDING_INDEPENDENT_REVIEW_AND_GO"
        ),
        "publisher": "ROOT_AUTHORITY_APPEND_ONLY_ONLY",
    },
    "REVIEW": {
        "artifact_type": "TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_REVIEW_V5",
        "status": "PASS_V5_EXECUTION_REVIEW_PENDING_INDEPENDENT_GO",
        "publisher": "INDEPENDENT_REVIEWER_APPEND_ONLY_ONLY",
    },
    "GO": {
        "artifact_type": "TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_GO_V5",
        "status": "GO_V5_SINGLE_TARGET_FREE_ATTEMPT",
        "publisher": "INDEPENDENT_GO_REVIEWER_APPEND_ONLY_ONLY",
    },
    "POSTRUN": {
        "artifact_type": "TRACK_A_TARGET_FREE_DUPLICATE_POSTRUN_PASS_V5",
        "status": "PASS_V5_NO_FIT_POSTRUN_AUDIT_PENDING_FINAL_SEAL",
        "publisher": "INDEPENDENT_POSTRUN_REVIEWER_APPEND_ONLY_ONLY",
    },
}
CONTROL_OUTPUT_NAMESPACE: Final[Mapping[str, Any]] = {
    "root": "modeling/target_free_multiseason_v5/",
    "runner_transaction_root": (
        "modeling/target_free_multiseason_v5/.transaction_v5/"
    ),
    "require_absent_at_go": True,
    "single_attempt_only": True,
}
DOCUMENTARY_AUTHORITY_SCOPE: Final[Mapping[str, Any]] = {
    "documentary_only": True,
    "execution_authorized": False,
    "value_read_authorized": False,
    "fit_authorized": False,
    "output_publication_authorized": False,
    "network_authorized": False,
    "label_or_future_year_authorized": False,
    "submission_csv_authorized": False,
}
EXECUTION_AUTHORITY_SCOPE: Final[Mapping[str, Any]] = {
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
GO_AUTHORITY_SCOPE: Final[Mapping[str, Any]] = {
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
POSTRUN_AUTHORITY_SCOPE: Final[Mapping[str, Any]] = {
    "documentary_only": True,
    "audit_pass_attested": True,
    "sealer_publication_condition_satisfied": True,
    "execution_authorized": False,
    "value_read_authorized": False,
    "fit_authorized": False,
    "network_authorized": False,
    "submission_csv_authorized": False,
}
EXECUTION_BUDGET: Final[Mapping[str, int]] = {
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
INDEPENDENT_CHECK_KEYS: Final[tuple[str, ...]] = (
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
REQUIRED_NEXT_CONTROLS: Final[Mapping[str, str]] = {
    "authorization": EXECUTION_AUTHORIZATION_RELATIVE_PATH,
    "independent_review": INDEPENDENT_REVIEW_RELATIVE_PATH,
    "independent_go": INDEPENDENT_GO_RELATIVE_PATH,
    "postrun_pass": POSTRUN_PASS_RELATIVE_PATH,
}
REQUIRED_REVIEW: Final[Mapping[str, str]] = {
    "path": INDEPENDENT_REVIEW_RELATIVE_PATH,
    "status_required": "PASS_V5_EXECUTION_REVIEW_PENDING_INDEPENDENT_GO",
}
REQUIRED_GO: Final[Mapping[str, str]] = {
    "path": INDEPENDENT_GO_RELATIVE_PATH,
    "status_required": "GO_V5_SINGLE_TARGET_FREE_ATTEMPT",
}
SINGLE_ATTEMPT: Final[Mapping[str, Any]] = {
    "attempt_number": 1,
    "output_root_was_absent_at_go": True,
    "retry_allowed": False,
    "prior_v5_failure_incident": None,
    "manifest_was_absent_at_go": True,
}
CONTROL_COMMAND_KEYS: Final[tuple[str, ...]] = ("runner", "auditor", "sealer")
TEST_EVIDENCE_KEYS: Final[tuple[str, ...]] = (
    "commands",
    "results",
    "all_exit_codes_zero",
    "all_expected_test_files_covered",
    "network_denied",
    "pycache_disabled",
    "code_identities_revalidated",
    "completed_utc",
)
TEST_RESULT_KEYS: Final[tuple[str, ...]] = (
    "test_file",
    "exit_code",
    "passed",
    "skipped",
    "deselected",
    "summary",
)
TEST_FILES: Final[Mapping[str, str]] = {
    "runner": "tests/test_noaa_gfs_target_free_duplicate_v5.py",
    "auditor": "tests/test_audit_noaa_gfs_target_free_duplicate_v5.py",
    "sealer": "tests/test_seal_noaa_gfs_target_free_duplicate_v5.py",
}
TEST_SUMMARY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<passed>[0-9]+) passed"
    r"(?:, (?P<skipped>[0-9]+) skipped)?"
    r"(?:, (?P<deselected>[0-9]+) deselected)?"
    r" in (?P<duration>[0-9]+(?:\.[0-9]+)?)s"
    r"(?: \((?P<hours>[0-9]+):(?P<minutes>[0-5][0-9]):"
    r"(?P<seconds>[0-5][0-9])\))?$"
)
PROCESS_CAPTURE_KEYS: Final[tuple[str, ...]] = (
    "command",
    "exit_code",
    "stdout_payload",
    "stdout_size_bytes",
    "stdout_sha256",
    "stderr_size_bytes",
    "stderr_sha256",
    "wall_seconds_observed",
    "full_stdout_identity_instrumented",
)
POSTRUN_AUDIT_RESULT_KEYS: Final[tuple[str, ...]] = (
    *PROCESS_CAPTURE_KEYS,
    "runner_result",
)
ATTEMPT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^target_free_duplicate_v5__[0-9]{8}T[0-9]{12}Z$"
)
FORBIDDEN_ACCESS_ATTESTATION: Final[Mapping[str, Any]] = {
    "labels_read": False,
    "arrays_2024_2025_read": False,
    "network_requests": 0,
    "generation_model_fit_units": 0,
    "submission_csv_files_written": 0,
}
FAMILY_DECISION_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "artifact_type",
    "status",
    "raw_component_order",
    "raw_component_results",
    "endpoint_veto_results",
    "global_ambiguity",
    "global_ambiguity_reasons",
    "family_clear_pass_results",
    "fixed_priority",
    "selected_family",
    "selected_raw_columns",
    "selected_derived_columns",
    "all_threshold_float_hex_witnesses",
    "all_tied_witnesses",
    "no_label_network_future_year_model_submission_attestation",
)
FAMILY_DECISION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "PASS_ONE_TARGET_FREE_FAMILY_SELECTED_PENDING_INDEPENDENT_POSTRUN_SEAL",
        "STOP_FAMILY_AMBIGUOUS",
        "STOP_NO_TARGET_FREE_INCREMENTAL_FAMILY",
    }
)
RAW_COMPONENT_RESULT_KEYS: Final[tuple[str, ...]] = (
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
ENDPOINT_VETO_RESULT_KEYS: Final[tuple[str, ...]] = (
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
THRESHOLD_WITNESS_KEYS: Final[tuple[str, ...]] = (
    "cutoff_name",
    "scope",
    "diagnostic",
    "value",
    "value_float_hex",
    "cutoff_float_hex",
    "boundary_equal",
)
TIED_WITNESS_KEYS: Final[tuple[str, ...]] = (
    "tie_kind",
    "diagnostic",
    "estimators",
    "metric_name",
    "metric_values_float_hex",
    "resolution",
    "global_ambiguity",
)
RAW_CLASSIFICATIONS: Final[tuple[str, ...]] = (
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
PREMODEL_CLASSIFICATIONS: Final[frozenset[str]] = frozenset(
    {
        "CLEAR_PHYSICAL_OR_COVERAGE_FAILURE",
        "CLEAR_NONNOVEL_CONSTANT",
    }
)
AMBIGUOUS_CLASSIFICATIONS: Final[frozenset[str]] = frozenset(
    {
        "AMBIGUOUS_THRESHOLD_EQUALITY",
        "AMBIGUOUS_PARENT_ESTIMATOR_TIE",
        "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE",
        "AMBIGUOUS_GRAY_ZONE",
    }
)
ENDPOINT_VERDICTS: Final[frozenset[str]] = frozenset(
    {
        "PASS_ENDPOINT_NONREDUNDANT_MARGIN",
        "CLEAR_ENDPOINT_VETO_FAILURE",
        "AMBIGUOUS_THRESHOLD_EQUALITY",
        "AMBIGUOUS_RAW_ESTIMATOR_TIE",
    }
)
CUTOFF_NAMES: Final[tuple[str, ...]] = (
    "AFFINE_ABS_PEARSON_0.9999",
    "AFFINE_ABS_SPEARMAN_0.9999",
    "AFFINE_NRMSE_0.01",
    "PARENT_R2_0.98",
    "PARENT_NRMSE_0.10",
    "INDEPENDENT_R2_EXACT_DUPLICATE_0.995",
    "INDEPENDENT_NRMSE_EXACT_DUPLICATE_0.02",
    "INDEPENDENT_R2_STABLE_MARGIN_0.9925",
    "INDEPENDENT_NRMSE_STABLE_MARGIN_0.025",
    "CELL_R2_0.995",
    "CELL_NRMSE_0.02",
    "CELL_WEAK_FLOOR_NRMSE_0.01",
    "MONTHLY_IQR_RATIO_0.1",
    "MONTHLY_IQR_RATIO_10",
    "YEAR_PSI_0.5",
)
CUTOFF_VALUES: Final[Mapping[str, float]] = {
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
TIE_KINDS: Final[tuple[str, ...]] = (
    "INDEPENDENT_RMSE",
    "PARENT_MAX_R2",
)
TIE_RESOLUTIONS: Final[frozenset[str]] = frozenset(
    {
        "GLOBAL_AMBIGUITY",
        "SHARED_PARENT_REDUNDANCY_TRUE",
        "SHARED_PARENT_REDUNDANCY_FALSE",
        "MASKED_BY_PRIOR_AFFINE_REDUNDANCY",
    }
)
FIXED_FAMILY_PRIORITY: Final[tuple[str, ...]] = (
    "LOW_LEVEL_ISOBARIC_WIND_PROFILE",
    "PBL_HEIGHT",
)


def make_threshold_witness(
    *,
    cutoff_name: str,
    scope: str,
    diagnostic: str,
    value: Any,
    cutoff: float,
) -> dict[str, Any]:
    if cutoff_name not in CUTOFF_NAMES:
        raise IntegrityError(f"unknown cutoff name: {cutoff_name}")
    if not _equal(cutoff, CUTOFF_VALUES[cutoff_name]):
        raise IntegrityError("cutoff argument differs from frozen cutoff name")
    decimal, witness = persist_float(value)
    return {
        "cutoff_name": cutoff_name,
        "scope": scope,
        "diagnostic": diagnostic,
        "value": decimal,
        "value_float_hex": witness,
        "cutoff_float_hex": canonical_float_hex(np_float64(cutoff)),
        "boundary_equal": _equal(value, cutoff),
    }


def np_float64(value: Any) -> Any:
    return _np().float64(value)


def _ordered_unique_estimators(values: Any, *, allow_empty: bool) -> list[str]:
    if not isinstance(values, list):
        raise IntegrityError("estimator witness must be a list")
    allowed = ("RIDGE_PIPELINE", "EXTRA_TREES")
    if any(value not in allowed for value in values):
        raise IntegrityError("estimator witness contains an unknown value")
    if len(values) != len(set(values)):
        raise IntegrityError("estimator witness contains a duplicate")
    if values != [value for value in allowed if value in values]:
        raise IntegrityError("estimator witness order differs")
    if not allow_empty and len(values) not in (1, 2):
        raise IntegrityError("parent estimator witness must have length 1 or 2")
    return values


def _validate_raw_component_result(
    value: Any, expected_response: str
) -> None:
    record = _require_exact_keys(
        value, RAW_COMPONENT_RESULT_KEYS, f"raw result {expected_response}"
    )
    if record["response"] != expected_response:
        raise IntegrityError("raw component response order differs")
    classification = record["classification"]
    if classification not in RAW_CLASSIFICATIONS:
        raise IntegrityError("raw component classification differs")
    if record["counts_as_nonredundant"] is not (
        classification == "PASS_NONREDUNDANT_MARGIN"
    ):
        raise IntegrityError("counts_as_nonredundant relation differs")
    no_selector = classification in PREMODEL_CLASSIFICATIONS or (
        classification == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
    )
    independent = record["independent_estimator"]
    if no_selector:
        if independent is not None:
            raise IntegrityError("independent estimator must be null")
    elif independent not in {"RIDGE_PIPELINE", "EXTRA_TREES"}:
        raise IntegrityError("independent estimator enum differs")
    parents = _ordered_unique_estimators(
        record["parent_reference_estimators"], allow_empty=no_selector
    )
    if no_selector and parents:
        raise IntegrityError("parent witnesses must be empty without selector")
    ambiguity = classification in AMBIGUOUS_CLASSIFICATIONS
    nullable_booleans = (
        "parent_redundancy_boolean",
        "cell_stability_pass",
        "distribution_stability_pass",
    )
    for key in nullable_booleans:
        observed = record[key]
        special_masked_parent = (
            key == "parent_redundancy_boolean"
            and classification == "CLEAR_REDUNDANT_AFFINE"
            and observed is None
        )
        if classification in PREMODEL_CLASSIFICATIONS or ambiguity:
            if observed is not None:
                raise IntegrityError(f"{key} must be null for unresolved record")
        elif not isinstance(observed, bool) and not special_masked_parent:
            raise IntegrityError(f"{key} must be boolean")
    predictors = record["affine_qualifying_predictors"]
    if not isinstance(predictors, list):
        raise IntegrityError("affine qualifying predictors must be a list")
    if len(predictors) != len(set(predictors)):
        raise IntegrityError("affine qualifying predictors are not unique")
    if predictors != [name for name in PREDICTORS if name in predictors]:
        raise IntegrityError("affine qualifying predictor order differs")
    if classification in PREMODEL_CLASSIFICATIONS or (
        classification == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
    ):
        if predictors:
            raise IntegrityError("affine predictors must be empty before scan")
    flags = record["boundary_equality_flags"]
    if not isinstance(flags, list) or flags != [
        name for name in CUTOFF_NAMES if name in flags
    ] or len(flags) != len(set(flags)):
        raise IntegrityError("boundary equality flags differ")
    if classification == "AMBIGUOUS_THRESHOLD_EQUALITY":
        if not flags:
            raise IntegrityError("threshold ambiguity requires a boundary flag")
    elif flags:
        raise IntegrityError("boundary flags only belong to threshold ambiguity")
    reason = record["ambiguity_reason"]
    if ambiguity:
        if reason != classification:
            raise IntegrityError("ambiguity reason must equal classification")
    elif reason is not None:
        raise IntegrityError("clear result ambiguity reason must be null")


def _endpoint_source_responses(diagnostic: str) -> tuple[str, ...]:
    if diagnostic == "ENDPOINT_DU_925_MINUS_1000":
        return ("UGRD_925mb", "UGRD_1000mb")
    if diagnostic == "ENDPOINT_DV_925_MINUS_1000":
        return ("VGRD_925mb", "VGRD_1000mb")
    if diagnostic == "ENDPOINT_VECTOR_SHEAR_MAG":
        return (
            "UGRD_925mb",
            "VGRD_925mb",
            "UGRD_1000mb",
            "VGRD_1000mb",
        )
    raise IntegrityError(f"unknown endpoint diagnostic: {diagnostic}")


def _validate_endpoint_result(value: Any, diagnostic: str) -> None:
    record = _require_exact_keys(
        value, ENDPOINT_VETO_RESULT_KEYS, f"endpoint result {diagnostic}"
    )
    if record["diagnostic"] != diagnostic:
        raise IntegrityError("endpoint result order differs")
    verdict = record["verdict"]
    if verdict not in ENDPOINT_VERDICTS:
        raise IntegrityError("endpoint verdict differs")
    binding = _require_exact_keys(
        record["source_component_estimator_binding"],
        _endpoint_source_responses(diagnostic),
        f"endpoint binding {diagnostic}",
    )
    for estimator in binding.values():
        if estimator not in {None, "RIDGE_PIPELINE", "EXTRA_TREES"}:
            raise IntegrityError("endpoint source estimator differs")
    source_null = any(estimator is None for estimator in binding.values())
    aggregate_keys = (
        "pooled_stable_margin_pass",
        "cell_stability_pass",
        "distribution_stability_pass",
    )
    aggregates = [record[key] for key in aggregate_keys]
    flags = record["boundary_equality_flags"]
    if not isinstance(flags, list) or flags != [
        name for name in CUTOFF_NAMES if name in flags
    ] or len(flags) != len(set(flags)):
        raise IntegrityError("endpoint boundary flag order differs")
    if source_null:
        if verdict not in {
            "CLEAR_ENDPOINT_VETO_FAILURE",
            "AMBIGUOUS_RAW_ESTIMATOR_TIE",
        }:
            raise IntegrityError("null endpoint source has incompatible verdict")
        if any(value is not None for value in aggregates) or flags:
            raise IntegrityError("unavailable endpoint metrics must be null")
    elif verdict == "AMBIGUOUS_THRESHOLD_EQUALITY":
        if not flags or not any(value is None for value in aggregates):
            raise IntegrityError("endpoint threshold ambiguity evidence differs")
        if any(value not in {None, True, False} for value in aggregates):
            raise IntegrityError("endpoint aggregate type differs")
    elif any(not isinstance(value, bool) for value in aggregates):
        raise IntegrityError("completed endpoint aggregates must be boolean")
    clear_pass = verdict == "PASS_ENDPOINT_NONREDUNDANT_MARGIN"
    if record["clear_pass"] is not clear_pass:
        raise IntegrityError("endpoint clear_pass relation differs")
    ambiguous = verdict in {
        "AMBIGUOUS_THRESHOLD_EQUALITY",
        "AMBIGUOUS_RAW_ESTIMATOR_TIE",
    }
    if ambiguous:
        if record["ambiguity_reason"] != verdict:
            raise IntegrityError("endpoint ambiguity reason differs")
    elif record["ambiguity_reason"] is not None:
        raise IntegrityError("clear endpoint ambiguity reason must be null")


def _scope_rank(scope: str, diagnostic: str) -> int:
    parts = scope.split("/")
    if parts[:2] == ["RAW", diagnostic] and len(parts) == 4:
        if parts[2] == "AFFINE" and parts[3] in PREDICTORS:
            return PREDICTORS.index(parts[3])
        if parts[2] in {"PARENT", "INDEPENDENT"} and parts[3] in {
            "RIDGE_PIPELINE",
            "EXTRA_TREES",
        }:
            return ("RIDGE_PIPELINE", "EXTRA_TREES").index(parts[3])
    if (
        len(parts) == 5
        and parts[0] in {"RAW", "ENDPOINT"}
        and parts[1] == diagnostic
        and parts[2] == "CELL"
    ):
        labels = {
            "YEAR": ("2022", "2023"),
            "SEASON": ("DJF", "MAM", "JJA", "SON"),
            "FORECAST_HOUR": ("EARLY", "MIDDLE", "LATE"),
            "GROUP": ("kpx_group_1", "kpx_group_2", "kpx_group_3"),
            "SITE": tuple(f"{index:02d}" for index in range(1, 18)),
        }
        if parts[3] in labels and parts[4] in labels[parts[3]]:
            base = {
                "YEAR": 0,
                "SEASON": 2,
                "FORECAST_HOUR": 6,
                "GROUP": 9,
                "SITE": 12,
            }[parts[3]]
            return base + labels[parts[3]].index(parts[4])
    if (
        len(parts) == 6
        and parts[0] in {"RAW", "ENDPOINT"}
        and parts[1] == diagnostic
        and parts[2] == "DISTRIBUTION"
    ):
        scopes = (
            "SITE_POOLED",
            "kpx_group_1",
            "kpx_group_2",
            "kpx_group_3",
        )
        if parts[3] in scopes:
            scope_index = scopes.index(parts[3])
            if parts[4] == "MONTH" and parts[5] in tuple(
                f"{month:02d}" for month in range(1, 13)
            ):
                return scope_index * 13 + int(parts[5]) - 1
            if parts[4:] == ["YEAR", "2022_2023"]:
                return scope_index * 13 + 12
    if (
        parts == ["ENDPOINT", diagnostic, "POOLED"]
        and diagnostic in ENDPOINT_VETO_DIAGNOSTICS
    ):
        return 0
    raise IntegrityError(f"threshold scope grammar differs: {scope}")


def _threshold_sort_key(record: Mapping[str, Any]) -> tuple[int, int, int]:
    diagnostics = (*RESPONSES, *ENDPOINT_VETO_DIAGNOSTICS)
    cutoff = str(record["cutoff_name"])
    diagnostic = str(record["diagnostic"])
    if cutoff not in CUTOFF_NAMES or diagnostic not in diagnostics:
        raise IntegrityError("threshold cutoff or diagnostic differs")
    return (
        CUTOFF_NAMES.index(cutoff),
        diagnostics.index(diagnostic),
        _scope_rank(str(record["scope"]), diagnostic),
    )


def validate_family_decision_document(document: Mapping[str, Any]) -> None:
    record = _require_exact_keys(
        document, FAMILY_DECISION_KEYS, "family decision"
    )
    if record["schema_version"] != SCHEMA_VERSION:
        raise IntegrityError("family decision schema_version differs")
    if record["artifact_type"] != "TARGET_FREE_FAMILY_DECISION_V5":
        raise IntegrityError("family decision artifact_type differs")
    if record["status"] not in FAMILY_DECISION_STATUSES:
        raise IntegrityError("family decision status differs")
    if record["raw_component_order"] != list(RESPONSES):
        raise IntegrityError("raw component order differs")
    raw_results = record["raw_component_results"]
    if not isinstance(raw_results, list) or len(raw_results) != 9:
        raise IntegrityError("raw component result count differs")
    for result, response in zip(raw_results, RESPONSES, strict=True):
        _validate_raw_component_result(result, response)
    endpoint_results = record["endpoint_veto_results"]
    if not isinstance(endpoint_results, list) or len(endpoint_results) != 3:
        raise IntegrityError("endpoint result count differs")
    for result, diagnostic in zip(
        endpoint_results, ENDPOINT_VETO_DIAGNOSTICS, strict=True
    ):
        _validate_endpoint_result(result, diagnostic)
    if record["fixed_priority"] != list(FIXED_FAMILY_PRIORITY):
        raise IntegrityError("fixed family priority differs")
    family_clear = _require_exact_keys(
        record["family_clear_pass_results"],
        FIXED_FAMILY_PRIORITY,
        "family clear pass results",
    )
    if any(not isinstance(value, bool) for value in family_clear.values()):
        raise IntegrityError("family clear pass result type differs")

    thresholds = record["all_threshold_float_hex_witnesses"]
    if not isinstance(thresholds, list):
        raise IntegrityError("threshold witnesses must be a list")
    threshold_keys: set[tuple[str, str, str]] = set()
    for witness in thresholds:
        item = _require_exact_keys(
            witness, THRESHOLD_WITNESS_KEYS, "threshold witness"
        )
        value = _finite_float64(item["value"], "threshold witness value")
        if item["value_float_hex"] != canonical_float_hex(value):
            raise IntegrityError("threshold value float hex differs")
        cutoff = CUTOFF_VALUES.get(item["cutoff_name"])
        if cutoff is None:
            raise IntegrityError("threshold cutoff name differs")
        if item["cutoff_float_hex"] != canonical_float_hex(np_float64(cutoff)):
            raise IntegrityError("threshold cutoff float hex differs")
        if not isinstance(item["boundary_equal"], bool):
            raise IntegrityError("threshold boundary flag type differs")
        if item["boundary_equal"] is not _equal(value, cutoff):
            raise IntegrityError("threshold direct equality flag differs")
        key = (item["cutoff_name"], item["scope"], item["diagnostic"])
        if key in threshold_keys:
            raise IntegrityError("duplicate threshold witness")
        threshold_keys.add(key)
    if thresholds != sorted(thresholds, key=_threshold_sort_key):
        raise IntegrityError("threshold witness order differs")

    ties = record["all_tied_witnesses"]
    if not isinstance(ties, list):
        raise IntegrityError("tied witnesses must be a list")
    tie_keys: set[tuple[str, str]] = set()
    tie_order: list[tuple[int, int]] = []
    for witness in ties:
        item = _require_exact_keys(witness, TIED_WITNESS_KEYS, "tied witness")
        if item["tie_kind"] not in TIE_KINDS:
            raise IntegrityError("tie kind differs")
        if item["diagnostic"] not in RESPONSES:
            raise IntegrityError("tie diagnostic differs")
        estimators = _ordered_unique_estimators(
            item["estimators"], allow_empty=False
        )
        if len(estimators) != 2:
            raise IntegrityError("tie must contain both estimators")
        expected_metric = (
            "RMSE"
            if item["tie_kind"] == "INDEPENDENT_RMSE"
            else "R2"
        )
        if item["metric_name"] != expected_metric:
            raise IntegrityError("tie metric differs")
        hexes = item["metric_values_float_hex"]
        if not isinstance(hexes, list) or len(hexes) != len(estimators):
            raise IntegrityError("tie metric witness cardinality differs")
        if any(
            not isinstance(value, str)
            or re.fullmatch(r"-?0x[0-9a-f]+\.[0-9a-f]+p[+-][0-9]+", value)
            is None
            for value in hexes
        ):
            raise IntegrityError("tie metric float hex differs")
        if item["resolution"] not in TIE_RESOLUTIONS:
            raise IntegrityError("tie resolution differs")
        if not isinstance(item["global_ambiguity"], bool):
            raise IntegrityError("tie global ambiguity type differs")
        key = (item["tie_kind"], item["diagnostic"])
        if key in tie_keys:
            raise IntegrityError("duplicate tied witness")
        tie_keys.add(key)
        tie_order.append(
            (
                RESPONSES.index(item["diagnostic"]),
                TIE_KINDS.index(item["tie_kind"]),
            )
        )
    if tie_order != sorted(tie_order):
        raise IntegrityError("tied witness order differs")
    raw_for_ties = {item["response"]: item for item in raw_results}
    for item in ties:
        raw_item = raw_for_ties[item["diagnostic"]]
        resolution = item["resolution"]
        if item["tie_kind"] == "INDEPENDENT_RMSE":
            if (
                resolution != "GLOBAL_AMBIGUITY"
                or item["global_ambiguity"] is not True
                or raw_item["classification"]
                != "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
            ):
                raise IntegrityError("independent tie resolution relation differs")
        elif resolution == "MASKED_BY_PRIOR_AFFINE_REDUNDANCY":
            if (
                item["global_ambiguity"] is not False
                or raw_item["classification"] != "CLEAR_REDUNDANT_AFFINE"
                or raw_item["parent_redundancy_boolean"] is not None
                or len(raw_item["parent_reference_estimators"]) != 2
                or not raw_item["affine_qualifying_predictors"]
            ):
                raise IntegrityError("masked parent tie relation differs")
        elif resolution == "GLOBAL_AMBIGUITY":
            if (
                item["global_ambiguity"] is not True
                or raw_item["classification"]
                != "AMBIGUOUS_PARENT_ESTIMATOR_TIE"
            ):
                raise IntegrityError("global parent tie relation differs")
        elif resolution == "SHARED_PARENT_REDUNDANCY_TRUE":
            if (
                item["global_ambiguity"] is not False
                or raw_item["parent_redundancy_boolean"] is not True
            ):
                raise IntegrityError("shared true parent tie relation differs")
        elif resolution == "SHARED_PARENT_REDUNDANCY_FALSE":
            if (
                item["global_ambiguity"] is not False
                or raw_item["parent_redundancy_boolean"] is not False
            ):
                raise IntegrityError("shared false parent tie relation differs")

    ambiguous_raw = any(
        item["classification"] in AMBIGUOUS_CLASSIFICATIONS
        for item in raw_results
    )
    ambiguous_endpoint = any(
        item["verdict"]
        in {"AMBIGUOUS_THRESHOLD_EQUALITY", "AMBIGUOUS_RAW_ESTIMATOR_TIE"}
        for item in endpoint_results
    )
    global_ambiguity = ambiguous_raw or ambiguous_endpoint
    if record["global_ambiguity"] is not global_ambiguity:
        raise IntegrityError("global ambiguity relation differs")
    reasons = record["global_ambiguity_reasons"]
    if not isinstance(reasons, list) or len(reasons) != len(set(reasons)):
        raise IntegrityError("global ambiguity reasons differ")
    if global_ambiguity != bool(reasons):
        raise IntegrityError("global ambiguity reason cardinality differs")
    expected_reasons: list[str] = []
    for item in raw_results:
        if item["classification"] == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE":
            expected_reasons.append(
                "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE:" + item["response"]
            )
    for witness in thresholds:
        if witness["boundary_equal"]:
            expected_reasons.append(
                "AMBIGUOUS_THRESHOLD_EQUALITY:"
                + witness["cutoff_name"]
                + ":"
                + witness["scope"]
                + ":"
                + witness["diagnostic"]
                + ":"
                + witness["value_float_hex"]
            )
    for item in raw_results:
        if item["classification"] == "AMBIGUOUS_PARENT_ESTIMATOR_TIE":
            expected_reasons.append(
                "AMBIGUOUS_PARENT_ESTIMATOR_TIE:" + item["response"]
            )
    for item in raw_results:
        if item["classification"] == "AMBIGUOUS_GRAY_ZONE":
            expected_reasons.append("AMBIGUOUS_GRAY_ZONE:" + item["response"])
    if reasons != expected_reasons:
        raise IntegrityError("global ambiguity reason derivation or order differs")
    if record[
        "no_label_network_future_year_model_submission_attestation"
    ] != dict(FORBIDDEN_ACCESS_ATTESTATION):
        raise IntegrityError("forbidden access attestation differs")

    raw_by_response = {item["response"]: item for item in raw_results}
    for endpoint in endpoint_results:
        binding = endpoint["source_component_estimator_binding"]
        for response, observed_estimator in binding.items():
            if observed_estimator != raw_by_response[response][
                "independent_estimator"
            ]:
                raise IntegrityError(
                    "endpoint binding differs from raw independent selector"
                )
    pbl_derived_pass = (
        raw_by_response["HPBL_surface"]["classification"]
        == "PASS_NONREDUNDANT_MARGIN"
    )
    passed_wind = [
        response
        for response in WIND_RESPONSES
        if raw_by_response[response]["classification"]
        == "PASS_NONREDUNDANT_MARGIN"
    ]
    passed_components = {name.split("_", 1)[0] for name in passed_wind}
    passed_levels = {int(name.split("_", 1)[1][:-2]) for name in passed_wind}
    wind_derived_pass = bool(
        all(
            raw_by_response[response]["classification"]
            != "CLEAR_PHYSICAL_OR_COVERAGE_FAILURE"
            for response in WIND_RESPONSES
        )
        and len(passed_wind) >= 4
        and {"UGRD", "VGRD"}.issubset(passed_components)
        and len(passed_levels) >= 2
        and all(item["clear_pass"] for item in endpoint_results)
    )
    if family_clear != {
        "LOW_LEVEL_ISOBARIC_WIND_PROFILE": wind_derived_pass,
        "PBL_HEIGHT": pbl_derived_pass,
    }:
        raise IntegrityError("family clear pass derivation differs")

    selected = record["selected_family"]
    raw_columns = record["selected_raw_columns"]
    derived_columns = record["selected_derived_columns"]
    if global_ambiguity:
        if (
            record["status"] != "STOP_FAMILY_AMBIGUOUS"
            or selected is not None
            or raw_columns != []
            or derived_columns != []
        ):
            raise IntegrityError("ambiguous terminal fields differ")
    elif selected == "LOW_LEVEL_ISOBARIC_WIND_PROFILE":
        if (
            record["status"]
            != "PASS_ONE_TARGET_FREE_FAMILY_SELECTED_PENDING_INDEPENDENT_POSTRUN_SEAL"
            or raw_columns != list(WIND_RESPONSES)
            or derived_columns != list(DERIVED_DIAGNOSTICS)
            or family_clear["LOW_LEVEL_ISOBARIC_WIND_PROFILE"] is not True
        ):
            raise IntegrityError("wind selected terminal fields differ")
    elif selected == "PBL_HEIGHT":
        if (
            record["status"]
            != "PASS_ONE_TARGET_FREE_FAMILY_SELECTED_PENDING_INDEPENDENT_POSTRUN_SEAL"
            or raw_columns != list(PBL_RESPONSES)
            or derived_columns != []
            or family_clear["PBL_HEIGHT"] is not True
            or family_clear["LOW_LEVEL_ISOBARIC_WIND_PROFILE"] is not False
        ):
            raise IntegrityError("PBL selected terminal fields differ")
    elif (
        selected is not None
        or record["status"] != "STOP_NO_TARGET_FREE_INCREMENTAL_FAMILY"
        or raw_columns != []
        or derived_columns != []
    ):
        raise IntegrityError("clear-no-family terminal fields differ")


class ControlValidator(Protocol):
    def __call__(self, artifact_root: Path) -> ControlValidation: ...


def _require_exact_keys(
    value: Any, keys: Sequence[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PredataAuthorizationError(f"{label} must be an object")
    expected = set(keys)
    if set(value) != expected or len(value) != len(expected):
        raise PredataAuthorizationError(
            f"{label} key set differs: {sorted(value)!r}"
        )
    return value


def _identity_mapping(path: Path, relative_path: str) -> dict[str, Any]:
    size, digest = _sha256_and_size(path)
    return {
        "path": relative_path,
        "size_bytes": size,
        "sha256": digest,
    }


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _expected_amendment_identity() -> dict[str, Any]:
    return {
        "path": AMENDMENT_RELATIVE_PATH,
        "size_bytes": AMENDMENT_SIZE_BYTES,
        "sha256": AMENDMENT_SHA256,
        "canonical_size_bytes": AMENDMENT_CANONICAL_SIZE_BYTES,
        "canonical_sha256": AMENDMENT_CANONICAL_SHA256,
    }


def _expected_amendment_v4_identity() -> dict[str, Any]:
    return {
        "path": AMENDMENT_V4_RELATIVE_PATH,
        "size_bytes": AMENDMENT_V4_SIZE_BYTES,
        "sha256": AMENDMENT_V4_SHA256,
        "canonical_size_bytes": AMENDMENT_V4_CANONICAL_SIZE_BYTES,
        "canonical_sha256": AMENDMENT_V4_CANONICAL_SHA256,
    }


def _expected_amendment_v5_identity() -> dict[str, Any]:
    return {
        "path": AMENDMENT_V5_RELATIVE_PATH,
        "size_bytes": AMENDMENT_V5_SIZE_BYTES,
        "sha256": AMENDMENT_V5_SHA256,
        "canonical_size_bytes": AMENDMENT_V5_CANONICAL_SIZE_BYTES,
        "canonical_sha256": AMENDMENT_V5_CANONICAL_SHA256,
    }


def _expected_v3_failure_incident_identity() -> dict[str, Any]:
    return {
        "path": V3_FAILURE_INCIDENT_RELATIVE_PATH,
        "size_bytes": V3_FAILURE_INCIDENT_SIZE_BYTES,
        "sha256": V3_FAILURE_INCIDENT_SHA256,
        "canonical_size_bytes": V3_FAILURE_INCIDENT_CANONICAL_SIZE_BYTES,
        "canonical_sha256": V3_FAILURE_INCIDENT_CANONICAL_SHA256,
    }


def _rfc3339_100ns_ticks(value: Any, label: str) -> int:
    if not isinstance(value, str):
        raise PredataAuthorizationError(f"{label} timestamp must be a string")
    match = re.fullmatch(
        r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d{6})Z",
        value,
    )
    if match is None:
        raise PredataAuthorizationError(
            f"{label} must be RFC3339 UTC with exactly 6 fractional digits"
        )
    year, month, day, hour, minute, second = map(
        int, match.groups()[:6]
    )
    fraction = match.group(7) + "0"
    try:
        base = dt.datetime(
            year, month, day, hour, minute, second, tzinfo=dt.UTC
        )
    except ValueError as exc:
        raise PredataAuthorizationError(f"{label} timestamp is invalid") from exc
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
    whole_seconds = int((base - epoch).total_seconds())
    return whole_seconds * 10_000_000 + int(fraction)


def validate_attempt_id(value: Any) -> str:
    if not isinstance(value, str) or ATTEMPT_ID_PATTERN.fullmatch(value) is None:
        raise PredataAuthorizationError("attempt_id format differs")
    timestamp = value.removeprefix("target_free_duplicate_v5__")
    try:
        parsed = dt.datetime.strptime(timestamp, "%Y%m%dT%H%M%S%fZ")
    except ValueError as exc:
        raise PredataAuthorizationError("attempt_id calendar time is invalid") from exc
    if parsed.strftime("%Y%m%dT%H%M%S%fZ") != timestamp:
        raise PredataAuthorizationError("attempt_id round-trip differs")
    return value


def required_control_commands(artifact_root: Path) -> dict[str, list[str]]:
    root = artifact_root.resolve(strict=True)
    python = (
        root.parents[1] / ".venv" / "Scripts" / "python.exe"
    ).resolve(strict=True)
    authorization = (root / EXECUTION_AUTHORIZATION_RELATIVE_PATH).resolve(
        strict=False
    )
    go = (root / INDEPENDENT_GO_RELATIVE_PATH).resolve(strict=False)
    postrun = (root / POSTRUN_PASS_RELATIVE_PATH).resolve(strict=False)
    common = [
        str(python),
        "-B",
        "-m",
        "",
        "--root",
        str(root),
        "--authorization",
        str(authorization),
        "--independent-go",
        str(go),
    ]
    result: dict[str, list[str]] = {}
    modules = {
        "runner": "scripts.run_noaa_gfs_target_free_duplicate_v5",
        "auditor": "scripts.audit_noaa_gfs_target_free_duplicate_v5",
        "sealer": "scripts.seal_noaa_gfs_target_free_duplicate_v5",
    }
    for role in CONTROL_COMMAND_KEYS:
        command = list(common)
        command[3] = modules[role]
        if role == "sealer":
            command.extend(["--postrun-pass", str(postrun)])
        result[role] = command
    return result


def validate_actual_runner_command(
    actual_orig_argv: Any,
    authorized_runner_argv: Any,
) -> list[str]:
    """Bind the live production process to the frozen runner command."""

    if not (
        type(authorized_runner_argv) is list
        and all(type(token) is str for token in authorized_runner_argv)
    ):
        raise PredataAuthorizationError(
            "authorized runner argv must be an exact list of strings"
        )
    if not (
        len(authorized_runner_argv) == 10
        and authorized_runner_argv[1:4]
        == ["-B", "-m", "scripts.run_noaa_gfs_target_free_duplicate_v5"]
        and authorized_runner_argv[4] == "--root"
        and authorized_runner_argv[6] == "--authorization"
        and authorized_runner_argv[8] == "--independent-go"
    ):
        raise PredataAuthorizationError("authorized runner argv shape differs")
    for index, label in (
        (0, "interpreter"),
        (5, "artifact root"),
        (7, "authorization"),
        (9, "independent GO"),
    ):
        if not Path(authorized_runner_argv[index]).is_absolute():
            raise PredataAuthorizationError(
                f"authorized runner {label} path must be absolute"
            )
    if not (
        type(actual_orig_argv) is list
        and all(type(token) is str for token in actual_orig_argv)
    ):
        raise PredataAuthorizationError(
            "actual runner sys.orig_argv must be an exact list of strings"
        )
    if not _exact_tree_equal(actual_orig_argv, authorized_runner_argv):
        raise PredataAuthorizationError(
            "actual runner sys.orig_argv differs from the frozen authorized command"
        )
    return list(actual_orig_argv)


def expected_test_commands(artifact_root: Path) -> dict[str, list[str]]:
    root = artifact_root.resolve(strict=True)
    repository_root = root.parents[1]
    python = (repository_root / ".venv" / "Scripts" / "python.exe").resolve(
        strict=True
    )
    return {
        role: [
            str(python),
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str((repository_root / relative_path).resolve(strict=True)),
        ]
        for role, relative_path in TEST_FILES.items()
    }


def _load_control(
    artifact_root: Path, relative_path: str
) -> tuple[Mapping[str, Any], Path, dict[str, Any]]:
    identity = FileIdentity(relative_path, 0, "0" * 64)
    identity.validate_shape()
    try:
        _assert_no_linklike_relative(
            artifact_root, relative_path, "control path"
        )
    except IdentityError as exc:
        raise PredataAuthorizationError(str(exc)) from exc
    root = artifact_root.resolve(strict=True)
    path = (root / relative_path).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PredataAuthorizationError("control path escapes artifact root") from exc
    if not path.is_file():
        raise PredataAuthorizationError(f"control is not a file: {relative_path}")
    raw = path.read_bytes()
    value = strict_json_loads(raw)
    if not isinstance(value, dict):
        raise PredataAuthorizationError("control top level must be an object")
    if strict_json_dumps(value, pretty=True) != raw:
        raise PredataAuthorizationError(
            "control deterministic pretty JSON serialization differs"
        )
    return value, path, _identity_mapping(path, relative_path)


def _exact_tree_equal(observed: Any, expected: Any) -> bool:
    """Compare frozen JSON trees without Python bool/int/float coercion."""

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


def _require_exact_tree(observed: Any, expected: Any, label: str) -> None:
    if not _exact_tree_equal(observed, expected):
        raise PredataAuthorizationError(f"{label} differs")


def _validate_control_envelope(role: str, record: Mapping[str, Any]) -> None:
    _require_exact_keys(record, CONTROL_TOP_KEYS[role], f"{role} control")
    _require_exact_tree(record["schema_version"], 5, f"{role} schema_version")
    for key, expected in CONTROL_LITERALS[role].items():
        _require_exact_tree(record[key], expected, f"{role} {key}")
    _require_exact_tree(
        record["amendment"], _expected_amendment_identity(),
        f"{role} amendment identity",
    )
    _require_exact_tree(
        record["amendment_v4"], _expected_amendment_v4_identity(),
        f"{role} amendment_v4 identity",
    )
    _require_exact_tree(
        record["amendment_v5"], _expected_amendment_v5_identity(),
        f"{role} amendment_v5 identity",
    )
    _require_exact_tree(
        record["output_namespace"], dict(CONTROL_OUTPUT_NAMESPACE),
        f"{role} output namespace",
    )
    _rfc3339_100ns_ticks(record["created_utc"], f"{role}.created_utc")
    _require_exact_keys(
        record["code_identities"], CODE_IDENTITY_KEYS, f"{role}.code_identities"
    )
    for name in CODE_IDENTITY_KEYS:
        _require_exact_keys(
            record["code_identities"][name],
            IDENTITY_KEYS,
            f"{role}.code_identities.{name}",
        )
    expected_scope = {
        "CODE_SEAL": DOCUMENTARY_AUTHORITY_SCOPE,
        "AUTHORIZATION": EXECUTION_AUTHORITY_SCOPE,
        "REVIEW": DOCUMENTARY_AUTHORITY_SCOPE,
        "GO": GO_AUTHORITY_SCOPE,
        "POSTRUN": POSTRUN_AUTHORITY_SCOPE,
    }[role]
    _require_exact_tree(
        record["authority_scope"], dict(expected_scope),
        f"{role} authority scope",
    )
    if role in {"REVIEW", "POSTRUN"}:
        _require_exact_tree(record["verdict"], "PASS", f"{role} verdict")


def _validate_code_seal_nested(
    record: Mapping[str, Any], artifact_root: Path
) -> None:
    evidence = _require_exact_keys(
        record["test_evidence"], TEST_EVIDENCE_KEYS, "CODE_SEAL.test_evidence"
    )
    commands = _require_exact_keys(
        evidence["commands"], CONTROL_COMMAND_KEYS, "test_evidence.commands"
    )
    results = _require_exact_keys(
        evidence["results"], CONTROL_COMMAND_KEYS, "test_evidence.results"
    )
    expected_commands = expected_test_commands(artifact_root)
    for role in CONTROL_COMMAND_KEYS:
        _require_exact_tree(
            commands[role], expected_commands[role], "test command argv"
        )
        result = _require_exact_keys(
            results[role], TEST_RESULT_KEYS, f"test_evidence.results.{role}"
        )
        if (
            not isinstance(result["exit_code"], int)
            or isinstance(result["exit_code"], bool)
            or result["exit_code"] != 0
        ):
            raise PredataAuthorizationError("test evidence exit code is nonzero")
        for key in ("passed", "skipped", "deselected"):
            if (
                not isinstance(result[key], int)
                or isinstance(result[key], bool)
                or result[key] < 0
            ):
                raise PredataAuthorizationError("test result count type differs")
        if result["passed"] <= 0:
            raise PredataAuthorizationError("test passed count must be positive")
        if result["test_file"] != TEST_FILES[role]:
            raise PredataAuthorizationError("test result file differs")
        summary = result["summary"]
        summary_match = (
            TEST_SUMMARY_PATTERN.fullmatch(summary)
            if isinstance(summary, str)
            else None
        )
        if summary_match is None:
            raise PredataAuthorizationError("test result summary differs")
        summary_counts = {
            key: int(summary_match.group(key) or 0)
            for key in ("passed", "skipped", "deselected")
        }
        if summary_counts != {
            key: result[key] for key in ("passed", "skipped", "deselected")
        }:
            raise PredataAuthorizationError(
                "test result summary count crosslink differs"
            )
        duration_text = summary_match.group("duration")
        whole_text, separator, fractional_text = duration_text.partition(".")
        duration_floor = int(whole_text)
        clock_groups = tuple(
            summary_match.group(name)
            for name in ("hours", "minutes", "seconds")
        )
        if any(value is not None for value in clock_groups):
            assert all(value is not None for value in clock_groups)
            observed_clock_seconds = (
                int(clock_groups[0]) * 3600
                + int(clock_groups[1]) * 60
                + int(clock_groups[2])
            )
            allowed_clock_seconds = {duration_floor}
            if (
                separator
                and fractional_text
                and set(fractional_text) == {"0"}
                and duration_floor > 0
            ):
                allowed_clock_seconds.add(duration_floor - 1)
            canonical_clock = (
                str(observed_clock_seconds // 3600),
                f"{(observed_clock_seconds % 3600) // 60:02d}",
                f"{observed_clock_seconds % 60:02d}",
            )
            if (
                observed_clock_seconds < 60
                or observed_clock_seconds not in allowed_clock_seconds
                or clock_groups != canonical_clock
            ):
                raise PredataAuthorizationError(
                    "long test duration clock suffix differs"
                )
    for key in (
        "all_exit_codes_zero",
        "all_expected_test_files_covered",
        "network_denied",
        "pycache_disabled",
        "code_identities_revalidated",
    ):
        if evidence[key] is not True:
            raise PredataAuthorizationError(f"test evidence {key} is not true")
    completed_ticks = _rfc3339_100ns_ticks(
        evidence["completed_utc"], "CODE_SEAL.test_evidence.completed_utc"
    )
    seal_ticks = _rfc3339_100ns_ticks(
        record["created_utc"], "CODE_SEAL.created_utc"
    )
    if completed_ticks > seal_ticks:
        raise PredataAuthorizationError(
            "test evidence completion is after code seal creation"
        )
    _require_exact_tree(
        record["required_next_controls"], dict(REQUIRED_NEXT_CONTROLS),
        "required next control map",
    )


def _validate_physical_code_identities(
    artifact_root: Path, code_identities: Mapping[str, Any]
) -> None:
    repository_root = artifact_root.resolve(strict=True).parents[1]
    for role in CODE_IDENTITY_KEYS:
        bound = code_identities[role]
        if (
            type(bound["path"]) is not str
            or type(bound["size_bytes"]) is not int
            or type(bound["sha256"]) is not str
            or bound["size_bytes"] < 0
            or re.fullmatch(r"[0-9a-f]{64}", bound["sha256"]) is None
        ):
            raise PredataAuthorizationError(
                f"code identity scalar type/shape differs: {role}"
            )
        expected = repository_root / CODE_IDENTITY_FILES[role]
        _assert_no_linklike_absolute(expected, f"code identity {role}")
        expected = expected.resolve(strict=True)
        if bound["path"] != expected.as_posix():
            raise PredataAuthorizationError(
                f"code identity absolute path differs: {role}"
            )
        size, digest = _sha256_and_size(expected)
        if size != bound["size_bytes"] or digest != bound["sha256"]:
            raise PredataAuthorizationError(
                f"code identity physical bytes differ: {role}"
            )


def validate_execution_budget(value: Any) -> None:
    record = _require_exact_keys(
        value, tuple(EXECUTION_BUDGET), "execution budget"
    )
    if any(
        not isinstance(item, int) or isinstance(item, bool)
        for item in record.values()
    ) or record != dict(EXECUTION_BUDGET):
        raise PredataAuthorizationError("execution budget differs")


def validate_runner_zero_state(artifact_root: Path) -> None:
    root = artifact_root.resolve(strict=True)
    if _lexists(root / POSTRUN_PASS_RELATIVE_PATH):
        raise PredataAuthorizationError(
            "postrun PASS path must be absent before runner execution"
        )
    if _lexists(root / OUTPUT_ROOT_RELATIVE):
        raise PredataAuthorizationError("output namespace is not absent at GO")
    if _lexists(root / V5_FAILURE_INCIDENT_RELATIVE_PATH):
        raise PredataAuthorizationError(
            "future V5 failure incident must be absent before execution"
        )


def validate_v3_failed_root_exact2(
    artifact_root: Path,
    incident: Mapping[str, Any] | None = None,
) -> tuple[FileIdentity, FileIdentity]:
    """Prove that the consumed V3 root preserves only its two aliases."""

    root = artifact_root.resolve(strict=True)
    failed_root = root / V3_FAILED_OUTPUT_ROOT_RELATIVE
    _assert_no_linklike_absolute(failed_root, "V3 failed output root")
    if not failed_root.is_dir():
        raise PredataAuthorizationError("V3 failed output root is absent")
    observed_names = tuple(sorted(item.name for item in failed_root.iterdir()))
    expected_names = tuple(
        identity.path.rsplit("/", 1)[-1]
        for identity in V3_FAILED_ALIAS_IDENTITIES
    )
    if observed_names != tuple(sorted(expected_names)):
        raise PredataAuthorizationError("V3 failed root is not exact2")
    source_identities = (
        FileIdentity(
            "manifest_census_v1.json",
            4_523,
            "0269f6717b1110fafb6bba052c78824c616c648dbbdfe9f630947012c3f571a1",
        ),
        FileIdentity(
            "raw/RAW_RANGE_MANIFEST.parquet",
            3_812_043,
            "11849682fd1c65ca20cfcb3002d3708beccf10ef862bb0f03d054c2c030790d5",
        ),
    )
    for identity, source_identity in zip(
        V3_FAILED_ALIAS_IDENTITIES, source_identities, strict=True
    ):
        path = revalidate_file_identity(root, identity)
        source = revalidate_file_identity(root, source_identity)
        metadata = path.stat()
        source_metadata = source.stat()
        if (
            not path.is_file()
            or _linklike(path)
            or metadata.st_nlink != 1
            or source_metadata.st_nlink != 1
            or os.path.samefile(source, path)
            or (
                source_metadata.st_ino
                and metadata.st_ino
                and source_metadata.st_ino == metadata.st_ino
            )
        ):
            raise PredataAuthorizationError(
                "V3 failed source/alias physical closure differs"
            )
    if incident is not None:
        poststate = incident.get("postfailure_state")
        if not isinstance(poststate, dict):
            raise PredataAuthorizationError("incident postfailure state missing")
        if (
            poststate.get("output_root")
            != V3_FAILED_OUTPUT_ROOT_RELATIVE + "/"
            or poststate.get("output_root_entry_count") != 2
            or poststate.get("output_root_entries_exact_order")
            != list(expected_names)
            or poststate.get("transaction_root_present") is not False
            or poststate.get("no_other_output_root_entries") is not True
        ):
            raise PredataAuthorizationError(
                "incident V3 exact2 afterstate crosslink differs"
            )
        aliases = _require_exact_keys(
            poststate.get("aliases"),
            ("FIELD_CENSUS_LOCK", "PROVENANCE_LEDGER"),
            "incident postfailure aliases",
        )
        for name, alias_identity, source_identity in zip(
            ("FIELD_CENSUS_LOCK", "PROVENANCE_LEDGER"),
            V3_FAILED_ALIAS_IDENTITIES,
            source_identities,
            strict=True,
        ):
            evidence = aliases[name]
            if (
                not isinstance(evidence, dict)
                or evidence.get("alias_identity")
                != dataclasses.asdict(alias_identity)
                or evidence.get("source_identity")
                != dataclasses.asdict(source_identity)
                or evidence.get("source_and_alias_bytes_equal") is not True
                or evidence.get("source_and_alias_file_ids_distinct") is not True
                or evidence.get("source_and_alias_samefile") is not False
                or evidence.get("source_persistent_link_count") != 1
                or evidence.get("alias_persistent_link_count") != 1
            ):
                raise PredataAuthorizationError(
                    "incident source/alias separation crosslink differs"
                )
    return V3_FAILED_ALIAS_IDENTITIES


def validate_historical_v3_control_identities(
    artifact_root: Path, incident: Mapping[str, Any]
) -> Mapping[str, Mapping[str, Any]]:
    """Rehash the immutable V3 controls named by the failure incident."""

    controls = _require_exact_keys(
        incident.get("control_identities"),
        ("authorization", "code_seal", "independent_go", "independent_review"),
        "incident.control_identities",
    )
    role_order = ("code_seal", "authorization", "independent_review", "independent_go")
    parsed: dict[str, Mapping[str, Any]] = {}
    for role in role_order:
        record = _require_exact_keys(
            controls[role], IDENTITY_KEYS, f"incident.control_identities.{role}"
        )
        if (
            type(record["path"]) is not str
            or type(record["size_bytes"]) is not int
            or type(record["sha256"]) is not str
        ):
            raise PredataAuthorizationError("historical control identity type differs")
        path = revalidate_file_identity(
            artifact_root,
            FileIdentity(record["path"], record["size_bytes"], record["sha256"]),
        )
        raw = path.read_bytes()
        value = strict_json_loads(raw)
        if not isinstance(value, dict) or strict_json_dumps(value, pretty=True) != raw:
            raise PredataAuthorizationError(
                "historical V3 control serialization differs"
            )
        parsed[role] = value
    return parsed


def validate_control_chronology(
    amendment: Mapping[str, Any],
    amendment_v4: Mapping[str, Any],
    amendment_v5: Mapping[str, Any],
    incident: Mapping[str, Any],
    historical_v3_controls: Mapping[str, Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    if tuple(records) != ("CODE_SEAL", "AUTHORIZATION", "REVIEW", "GO"):
        raise PredataAuthorizationError("control chronology role order differs")
    v3_ticks = _rfc3339_100ns_ticks(
        amendment["created_utc"], "amendment.created_utc"
    )
    v4_ticks = _rfc3339_100ns_ticks(
        amendment_v4["created_utc"], "amendment_v4.created_utc"
    )
    incident_ticks = _rfc3339_100ns_ticks(
        incident["created_utc"], "V3 incident.created_utc"
    )
    v5_ticks = _rfc3339_100ns_ticks(
        amendment_v5["created_utc"], "amendment_v5.created_utc"
    )
    historical_ticks = [
        _rfc3339_100ns_ticks(
            historical_v3_controls[role]["created_utc"],
            f"historical V3 {role}.created_utc",
        )
        for role in ("code_seal", "authorization", "independent_review", "independent_go")
    ]
    completed_ticks = _rfc3339_100ns_ticks(
        records["CODE_SEAL"]["test_evidence"]["completed_utc"],
        "CODE_SEAL.test_evidence.completed_utc",
    )
    control_ticks = [
        _rfc3339_100ns_ticks(
            records[role]["created_utc"], f"{role}.created_utc"
        )
        for role in records
    ]
    seal_ticks = control_ticks[0]
    if (
        not (
            v3_ticks
            < v4_ticks
            < historical_ticks[0]
            < historical_ticks[1]
            < historical_ticks[2]
            < historical_ticks[3]
            < incident_ticks
            < v5_ticks
            < completed_ticks
            <= seal_ticks
        )
        or any(
            left >= right
            for left, right in zip(control_ticks, control_ticks[1:])
        )
    ):
        raise PredataAuthorizationError("control chronology is not strict")


def validate_execution_control_chain(
    artifact_root: Path,
    *,
    authorization_path: Path,
    independent_go_path: Path,
    actual_orig_argv: Any = None,
    require_actual_orig_argv: bool = False,
) -> ControlValidation:
    """Strictly validate CODE_SEAL -> AUTH -> REVIEW -> GO before aliases."""

    try:
        lexical_root = Path(artifact_root)
        if not lexical_root.is_absolute():
            raise PredataAuthorizationError(
                "artifact root CLI path must be absolute"
            )
        _assert_no_linklike_absolute(lexical_root, "artifact root CLI path")
        root = lexical_root.resolve(strict=True)
    except (IdentityError, OSError) as exc:
        raise PredataAuthorizationError(
            "artifact root CLI path is link-like, absent, or unreadable"
        ) from exc
    if _normcase_lexical(artifact_root) != _normcase_lexical(root):
        raise PredataAuthorizationError(
            "artifact root CLI path is not canonical lexical absolute"
        )
    expected_authorization = root / EXECUTION_AUTHORIZATION_RELATIVE_PATH
    expected_go = root / INDEPENDENT_GO_RELATIVE_PATH
    _validate_canonical_absolute_path(
        authorization_path, expected_authorization, label="authorization"
    )
    _validate_canonical_absolute_path(
        independent_go_path, expected_go, label="independent GO"
    )
    validate_runner_zero_state(root)
    amendment = validate_amendment(root)
    amendment_v4 = validate_amendment_v4(root, amendment)
    incident = validate_v3_failure_incident(root)
    amendment_v5 = validate_amendment_v5(
        root, amendment, amendment_v4, incident
    )
    validate_v3_failed_root_exact2(root, incident)
    historical_v3_controls = validate_historical_v3_control_identities(
        root, incident
    )

    seal, _, seal_identity = _load_control(root, CODE_SEAL_RELATIVE_PATH)
    auth, _, auth_identity = _load_control(
        root, EXECUTION_AUTHORIZATION_RELATIVE_PATH
    )
    review, _, review_identity = _load_control(
        root, INDEPENDENT_REVIEW_RELATIVE_PATH
    )
    go, _, _go_identity = _load_control(root, INDEPENDENT_GO_RELATIVE_PATH)
    records = {
        "CODE_SEAL": seal,
        "AUTHORIZATION": auth,
        "REVIEW": review,
        "GO": go,
    }
    for role, record in records.items():
        _validate_control_envelope(role, record)
    _validate_code_seal_nested(seal, root)
    common_code = seal["code_identities"]
    if any(
        not _exact_tree_equal(record["code_identities"], common_code)
        for record in records.values()
    ):
        raise PredataAuthorizationError("control code identity crosslink differs")
    _validate_physical_code_identities(root, common_code)

    _require_exact_tree(
        auth["code_seal"], seal_identity, "authorization code-seal crosslink"
    )
    _require_exact_tree(
        review["code_seal"], seal_identity, "review code-seal crosslink"
    )
    _require_exact_tree(
        review["authorization"], auth_identity, "review authorization crosslink"
    )
    _require_exact_tree(
        go["code_seal"], seal_identity, "GO code-seal crosslink"
    )
    _require_exact_tree(
        go["authorization"], auth_identity, "GO authorization crosslink"
    )
    _require_exact_tree(
        go["independent_review"], review_identity, "GO review crosslink"
    )

    attempt_id = validate_attempt_id(auth["attempt_id"])
    if (
        not _exact_tree_equal(review["attempt_id"], attempt_id)
        or not _exact_tree_equal(go["attempt_id"], attempt_id)
    ):
        raise PredataAuthorizationError("attempt_id crosslink differs")
    validate_execution_budget(auth["execution_budget"])
    commands = required_control_commands(root)
    if (
        not _exact_tree_equal(auth["required_commands"], commands)
        or not _exact_tree_equal(go["required_commands"], commands)
    ):
        raise PredataAuthorizationError("required command argv differs")
    if require_actual_orig_argv:
        validate_actual_runner_command(actual_orig_argv, commands["runner"])
    _require_exact_tree(
        auth["required_review"], dict(REQUIRED_REVIEW),
        "required review pointer",
    )
    _require_exact_tree(
        auth["required_go"], dict(REQUIRED_GO), "required GO pointer"
    )
    checks = _require_exact_keys(
        review["independent_checks"],
        INDEPENDENT_CHECK_KEYS,
        "REVIEW.independent_checks",
    )
    if any(checks[key] is not True for key in INDEPENDENT_CHECK_KEYS):
        raise PredataAuthorizationError("independent review check is not true")
    _require_exact_tree(
        go["single_attempt"], dict(SINGLE_ATTEMPT),
        "single attempt contract",
    )
    validate_control_chronology(
        amendment,
        amendment_v4,
        amendment_v5,
        incident,
        historical_v3_controls,
        records,
    )
    return ControlValidation(
        amendment_pass=True,
        code_seal_pass=True,
        execution_authorization_pass=True,
        independent_review_pass=True,
        independent_go_pass=True,
        runtime_identity_pass=False,
        bound_input_identity_and_schema_pass=False,
        output_namespace_absent_pass=True,
        aliases_pass=False,
    )


def revalidate_all_bound_inputs(
    artifact_root: Path, amendment: Mapping[str, Any]
) -> dict[str, Path]:
    """Rehash the exact 31 bound files without interpreting Parquet values."""

    root = artifact_root.resolve(strict=True)
    bound = _require_exact_keys(
        amendment.get("bound_inputs"),
        (
            "bounded_group_predictors",
            "bounded_predictor_audit",
            "bounded_predictor_builder",
            "bounded_predictor_causal_binding",
            "bounded_predictor_contract",
            "bounded_predictor_manifest",
            "bounded_site_predictors",
            "causal_validation_contract",
            "census_cumulative_access_amendment",
            "census_manifest",
            "census_raw_download_plan",
            "coordinate_lock",
            "decoded_group_matrix",
            "decoded_matrix_lock",
            "decoded_site_matrix",
            "independent_census_audit",
            "independent_target_free_preregister",
            "parent_amendment_v2",
            "parent_plan",
            "physical_and_coverage_audit",
            "raw_access_ledger",
            "raw_range_manifest_csv",
            "raw_range_manifest_parquet",
            "v8_auditor",
            "v8_auditor_test",
            "v8_authorization",
            "v8_independent_go",
            "v8_independent_review",
            "v8_live_success_attestation",
            "v8_sealer",
            "v8_sealer_test",
        ),
        "amendment.bound_inputs",
    )
    result: dict[str, Path] = {}
    for name, raw_identity in bound.items():
        if not isinstance(raw_identity, dict):
            raise IdentityError(f"bound identity is not an object: {name}")
        required = {"path", "size_bytes", "sha256"}
        if not required.issubset(raw_identity):
            raise IdentityError(f"bound identity shape differs: {name}")
        relative_or_absolute = str(raw_identity["path"])
        candidate = Path(relative_or_absolute)
        if candidate.is_absolute():
            _assert_no_linklike_absolute(candidate, f"bound input {name}")
            path = candidate.resolve(strict=True)
            if path.as_posix() != relative_or_absolute.replace("\\", "/"):
                raise IdentityError(f"absolute bound path differs: {name}")
        else:
            identity = FileIdentity(
                relative_or_absolute,
                int(raw_identity["size_bytes"]),
                str(raw_identity["sha256"]),
            )
            path = resolve_relative_identity(root, identity)
        size, digest = _sha256_and_size(path)
        if size != int(raw_identity["size_bytes"]) or digest != str(
            raw_identity["sha256"]
        ):
            raise IdentityError(f"bound input identity differs: {name}")
        if path.suffix.lower() == ".json":
            strict_json_loads(path.read_bytes())
        result[name] = path
    if len(result) != 31:
        raise AssertionError("bound input rehash count drift")
    return result


EXTERNAL_SITE_COLUMNS: Final[tuple[str, ...]] = (
    "valid_time_utc",
    "target_operating_day_kst",
    "run_init_utc",
    "forecast_hour",
    "site_id",
    "group",
    "latitude",
    "longitude",
    "capacity_mw",
    *RESPONSES,
)
EXTERNAL_GROUP_COLUMNS: Final[tuple[str, ...]] = (
    "valid_time_utc",
    "target_operating_day_kst",
    "run_init_utc",
    "forecast_hour",
    "group",
    "site_count",
    "capacity_mw",
    *RESPONSES,
)
PREDICTOR_SITE_COLUMNS: Final[tuple[str, ...]] = (
    *SITE_METADATA_COLUMNS,
    *PREDICTORS,
)
PREDICTOR_GROUP_COLUMNS: Final[tuple[str, ...]] = (
    *GROUP_METADATA_COLUMNS,
    *PREDICTORS,
)
PROVENANCE_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "source_archive",
    "archive_product",
    "run_init_utc",
    "forecast_hour",
    "valid_time_utc",
    "retrieval_url_or_request_id",
    "retrieved_at",
    "raw_filename",
    "raw_sha256",
    "official_metadata",
    "publication_evidence_type",
    "publication_evidence_reference",
    "cutoff_utc",
    "cutoff_margin",
    "status",
)
CENSUS_ALIAS_TOP_KEYS: Final[tuple[str, ...]] = (
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


def _validate_census_manifest_identity(
    artifact_root: Path, value: Any, label: str
) -> None:
    record = _require_exact_keys(value, IDENTITY_KEYS, label)
    raw_path = record["path"]
    size = record["size_bytes"]
    digest = record["sha256"]
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise IntegrityError(f"{label} identity shape differs")
    path_value = Path(raw_path)
    if path_value.is_absolute():
        _assert_no_linklike_absolute(path_value, label)
        candidate = path_value.resolve(strict=True)
        if candidate.as_posix() != raw_path.replace("\\", "/"):
            raise IntegrityError(f"{label} absolute path normalization differs")
    else:
        if "\\" in raw_path:
            raise IntegrityError(f"{label} relative path must use forward slash")
        _assert_no_linklike_relative(artifact_root, raw_path, label)
        root = artifact_root.resolve(strict=True)
        candidate = (root / path_value).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise IntegrityError(f"{label} relative path escapes root") from exc
    observed_size, observed_digest = _sha256_and_size(candidate)
    if observed_size != size or observed_digest != digest:
        raise IdentityError(f"{label} referenced identity differs")


def validate_census_alias_document(
    artifact_root: Path, alias_path: Path
) -> None:
    _assert_no_linklike_absolute(alias_path, "FIELD_CENSUS_LOCK alias")
    document = strict_json_loads(alias_path.read_bytes())
    record = _require_exact_keys(
        document, CENSUS_ALIAS_TOP_KEYS, "FIELD_CENSUS_LOCK alias"
    )
    if (
        record["schema_version"] != 1
        or record["artifact_type"]
        != "NOAA_GFS_MULTISEASON_CENSUS_MANIFEST"
        or record["labels_read"] is not False
        or record["models_fit"] != 0
        or record["raw_downloaded_bytes"] != 0
        or record["raw_network_requests"] != 0
        or record["submission_csv_created"] is not False
    ):
        raise IntegrityError("FIELD_CENSUS_LOCK semantic closure differs")
    _rfc3339_100ns_ticks(record["created_utc"], "census created_utc")
    identity_keys = (
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
    for key in identity_keys:
        nested_identity = _require_exact_keys(
            record[key], IDENTITY_KEYS, f"FIELD_CENSUS_LOCK.{key}"
        )
        manifest_path = Path(str(nested_identity["path"]))
        if key in {"reproduction_code", "test_code"}:
            if not manifest_path.is_absolute():
                raise IntegrityError(
                    f"FIELD_CENSUS_LOCK.{key} path must be absolute"
                )
        elif manifest_path.is_absolute():
            raise IntegrityError(
                f"FIELD_CENSUS_LOCK.{key} path must be artifact-relative"
            )
        _validate_census_manifest_identity(
            artifact_root, record[key], f"FIELD_CENSUS_LOCK.{key}"
        )
    checkpoints = record["progress_checkpoints"]
    if not isinstance(checkpoints, list) or len(checkpoints) != 13:
        raise IntegrityError("census progress checkpoint count differs")
    checkpoint_paths: list[str] = []
    for index, checkpoint in enumerate(checkpoints):
        checkpoint_identity = _require_exact_keys(
            checkpoint,
            IDENTITY_KEYS,
            f"FIELD_CENSUS_LOCK.progress_checkpoints[{index}]",
        )
        if Path(str(checkpoint_identity["path"])).is_absolute():
            raise IntegrityError("census checkpoint path must be relative")
        _validate_census_manifest_identity(
            artifact_root,
            checkpoint,
            f"FIELD_CENSUS_LOCK.progress_checkpoints[{index}]",
        )
        checkpoint_paths.append(str(checkpoint["path"]))
    if len(checkpoint_paths) != len(set(checkpoint_paths)):
        raise IntegrityError("census progress checkpoint path is duplicated")


def _validate_parquet_names_types_rows(
    authorization: PredataAuthorization,
    path: Path,
    *,
    expected_columns: Sequence[str],
    expected_type_names: Sequence[str],
    expected_rows: int,
) -> None:
    parquet_file = guarded_parquet_file(authorization, path)
    try:
        import pyarrow as pa

        constructors: Mapping[str, Callable[[], Any]] = {
            "large_string": pa.large_string,
            "int16": pa.int16,
            "int64": pa.int64,
            "float64": pa.float64,
        }
        schema = parquet_file.schema_arrow
        observed_columns = tuple(schema.names)
        if any(name not in constructors for name in expected_type_names):
            raise IntegrityError(f"Parquet expected type token differs: {path}")
        expected_types = tuple(
            constructors[name]() for name in expected_type_names
        )
        observed_types = tuple(field.type for field in schema)
        if observed_columns != tuple(expected_columns):
            raise IntegrityError(f"Parquet column order differs: {path}")
        if len(observed_types) != len(expected_types) or any(
            not observed.equals(expected)
            for observed, expected in zip(
                observed_types, expected_types, strict=True
            )
        ):
            raise IntegrityError(
                f"Parquet structural DataType vector differs: {path}"
            )
        if int(parquet_file.metadata.num_rows) != expected_rows:
            raise IntegrityError(f"Parquet row count differs: {path}")
    finally:
        parquet_file.close()


def _validate_provenance_alias_schema(
    authorization: PredataAuthorization,
    path: Path,
    *,
    metadata_only_before_handoff: bool = False,
) -> None:
    parquet_file = (
        guarded_alias_parquet_file(authorization, path)
        if metadata_only_before_handoff
        else guarded_parquet_file(authorization, path)
    )
    try:
        names = tuple(parquet_file.schema_arrow.names)
        if int(parquet_file.metadata.num_rows) != 10_368:
            raise IntegrityError("provenance alias row count differs")
        if len(names) != len(set(names)):
            raise IntegrityError("provenance alias contains duplicate columns")
        missing = set(PROVENANCE_REQUIRED_FIELDS).difference(names)
        if missing:
            raise IntegrityError(
                f"provenance alias lacks exact15 fields: {sorted(missing)!r}"
            )
    finally:
        parquet_file.close()


FOOTER_GATE_ROLES: Final[tuple[str, ...]] = (
    "decoded_site_matrix",
    "decoded_group_matrix",
    "bounded_site_predictors",
    "bounded_group_predictors",
)
FOOTER_ROLE_KEYS: Final[tuple[str, ...]] = (
    "bound_input_key",
    "column_names_exact",
    "pyarrow_constructor_types_exact",
    "row_count_exact",
    "source_identity",
    "structural_schema_fingerprint",
)


def _expected_footer_contracts() -> Mapping[str, tuple[tuple[str, ...], tuple[str, ...], int]]:
    return {
        "decoded_site_matrix": (
            EXTERNAL_SITE_COLUMNS,
            (
                "large_string",
                "large_string",
                "large_string",
                "int64",
                "int64",
                "large_string",
                *("float64" for _ in range(12)),
            ),
            EXPECTED_SITE_ROWS,
        ),
        "decoded_group_matrix": (
            EXTERNAL_GROUP_COLUMNS,
            (
                "large_string",
                "large_string",
                "large_string",
                "int64",
                "large_string",
                "int64",
                *("float64" for _ in range(10)),
            ),
            EXPECTED_GROUP_ROWS,
        ),
        "bounded_site_predictors": (
            PREDICTOR_SITE_COLUMNS,
            (
                "large_string",
                "large_string",
                "large_string",
                "large_string",
                "int16",
                "large_string",
                *("float64" for _ in range(38)),
            ),
            EXPECTED_SITE_ROWS,
        ),
        "bounded_group_predictors": (
            PREDICTOR_GROUP_COLUMNS,
            (
                "large_string",
                "large_string",
                "large_string",
                "large_string",
                "large_string",
                "int16",
                *("float64" for _ in range(36)),
            ),
            EXPECTED_GROUP_ROWS,
        ),
    }


def validate_pre_root_source_metadata_gate(
    artifact_root: Path,
    bound_paths: Mapping[str, Path],
    amendment: Mapping[str, Any],
    amendment_v5: Mapping[str, Any],
    *,
    authorization: PredataAuthorization,
) -> None:
    """Validate exact4 source footers and alias sources with handoff false."""

    root = artifact_root.resolve(strict=True)
    validate_runner_zero_state(root)
    validate_v3_failed_root_exact2(root)
    if authorization.arrow_handoff_started or authorization.alias_files_published:
        raise PredataAuthorizationError(
            "pre-root footer gate was entered after publication or handoff"
        )
    contracts = _require_exact_keys(
        amendment_v5.get("bound_parquet_footer_schemas_exact"),
        (
            "bounded_group_predictors",
            "bounded_site_predictors",
            "common_footer_requirements",
            "decoded_group_matrix",
            "decoded_site_matrix",
            "forbidden_acceptance_routes",
            "gate_roles_exact_order",
            "structural_schema_canonicalization",
        ),
        "V5 bound_parquet_footer_schemas_exact",
    )
    if contracts["gate_roles_exact_order"] != list(FOOTER_GATE_ROLES):
        raise IdentityError("V5 exact4 footer role order differs")
    expected_contracts = _expected_footer_contracts()
    bound_inputs = amendment.get("bound_inputs")
    if not isinstance(bound_inputs, dict) or any(
        role not in bound_inputs for role in FOOTER_GATE_ROLES
    ):
        raise IdentityError("V3 bound input footer roles are absent")
    allowed_paths = tuple(bound_paths[role] for role in FOOTER_GATE_ROLES) + (
        bound_paths["raw_range_manifest_parquet"],
    )
    authorization.source_metadata_scope_open = True
    try:
        import pyarrow as pa

        constructors: Mapping[str, Callable[[], Any]] = {
            "large_string": pa.large_string,
            "int16": pa.int16,
            "int64": pa.int64,
            "float64": pa.float64,
        }
        for role in FOOTER_GATE_ROLES:
            contract = _require_exact_keys(
                contracts[role], FOOTER_ROLE_KEYS, f"V5 footer role {role}"
            )
            expected_columns, expected_tokens, expected_rows = expected_contracts[role]
            if (
                contract["bound_input_key"] != role
                or contract["column_names_exact"] != list(expected_columns)
                or contract["pyarrow_constructor_types_exact"]
                != list(expected_tokens)
                or type(contract["row_count_exact"]) is not int
                or contract["row_count_exact"] != expected_rows
                or contract["source_identity"] != bound_inputs[role]
            ):
                raise IdentityError(f"V5 documentary footer contract differs: {role}")
            if any(token not in constructors for token in expected_tokens):
                raise IdentityError("V5 footer constructor token differs")
            expected_types = tuple(constructors[token]() for token in expected_tokens)
            parquet_file = guarded_source_parquet_file(
                authorization,
                bound_paths[role],
                allowed_paths=allowed_paths,
            )
            try:
                schema = parquet_file.schema_arrow
                rows = int(parquet_file.metadata.num_rows)
                observed_types = tuple(field.type for field in schema)
                if tuple(schema.names) != expected_columns or rows != expected_rows:
                    raise IntegrityError(f"pre-root footer names/rows differ: {role}")
                if len(observed_types) != len(expected_types) or any(
                    not observed.equals(expected)
                    for observed, expected in zip(
                        observed_types, expected_types, strict=True
                    )
                ):
                    raise IntegrityError(
                        f"pre-root structural DataType vector differs: {role}"
                    )
                witness = {
                    "columns": list(schema.names),
                    "rows": rows,
                    "types": [str(item) for item in observed_types],
                }
                witness_bytes = strict_json_dumps(witness, pretty=False)
                fingerprint = _require_exact_keys(
                    contract["structural_schema_fingerprint"],
                    ("canonical_sha256", "canonical_size_bytes"),
                    f"V5 footer fingerprint {role}",
                )
                if (
                    len(witness_bytes) != fingerprint["canonical_size_bytes"]
                    or hashlib.sha256(witness_bytes).hexdigest()
                    != fingerprint["canonical_sha256"]
                ):
                    raise IntegrityError(
                        f"pre-root structural fingerprint differs: {role}"
                    )
            finally:
                parquet_file.close()

        validate_census_alias_document(root, bound_paths["census_manifest"])
        provenance = guarded_source_parquet_file(
            authorization,
            bound_paths["raw_range_manifest_parquet"],
            allowed_paths=allowed_paths,
        )
        try:
            names = tuple(provenance.schema_arrow.names)
            if (
                int(provenance.metadata.num_rows) != 10_368
                or len(names) != len(set(names))
                or set(PROVENANCE_REQUIRED_FIELDS).difference(names)
            ):
                raise IntegrityError(
                    "pre-root provenance footer semantic closure differs"
                )
        finally:
            provenance.close()
    finally:
        authorization.source_metadata_scope_open = False
    validate_runner_zero_state(root)
    authorization.source_footer_metadata_validated = True
    authorization.validation = dataclasses.replace(
        authorization.validation, bound_input_identity_and_schema_pass=True
    )


@dataclasses.dataclass(frozen=True, slots=True)
class JoinedInputs:
    site: Any
    group: Any
    predictor_matrix_float64: Any
    fold_ordinals: Any


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedExecution:
    artifact_root: Path
    attempt_id: str
    amendment: Mapping[str, Any]
    amendment_v4: Mapping[str, Any]
    amendment_v5: Mapping[str, Any]
    v3_failure_incident: Mapping[str, Any]
    bound_paths: Mapping[str, Path]
    authorization: PredataAuthorization
    joined: JoinedInputs
    runtime_evidence: Mapping[str, Any]
    alias_identities: tuple[FileIdentity, ...]


def _validate_recomputed_physical_pass(joined: JoinedInputs) -> None:
    np = _np()
    site = joined.site
    hpbl = site["HPBL_surface"].to_numpy(dtype=np.float64, copy=True)
    wind = site.loc[:, WIND_RESPONSES].to_numpy(dtype=np.float64, copy=True)
    if (
        hpbl.shape != (EXPECTED_SITE_ROWS,)
        or wind.shape != (EXPECTED_SITE_ROWS, 8)
        or not bool(np.isfinite(hpbl).all())
        or not bool(np.isfinite(wind).all())
        or bool(np.any(hpbl < np.float64(0.0)))
        or bool(np.any(hpbl > np.float64(10_000.0)))
        or bool(np.any(np.abs(wind) > np.float64(150.0)))
    ):
        raise IntegrityError(
            "STOP_INDEPENDENT_AUDIT_MISMATCH_GLOBAL_INTEGRITY_FAILURE"
        )


def prepare_execution(
    artifact_root: Path,
    *,
    authorization_path: Path,
    independent_go_path: Path,
    actual_orig_argv: Any = None,
    require_actual_orig_argv: bool = False,
) -> PreparedExecution:
    """Pass controls/runtime/bytes, publish two aliases, then hand off to Arrow."""

    validation = validate_execution_control_chain(
        artifact_root,
        authorization_path=authorization_path,
        independent_go_path=independent_go_path,
        actual_orig_argv=actual_orig_argv,
        require_actual_orig_argv=require_actual_orig_argv,
    )
    root = artifact_root.resolve(strict=True)
    authorization_document, _, _ = _load_control(
        root, EXECUTION_AUTHORIZATION_RELATIVE_PATH
    )
    attempt_id = validate_attempt_id(authorization_document["attempt_id"])
    runtime_evidence = validate_runtime_identity()
    recorder = ThreadpoolCaptureRecorder()
    recorder.capture(
        phase="BEFORE_EXPLICIT_CONTEXT",
        label="RUNTIME_IDENTITY_VALIDATED_BEFORE_ANY_FIT_OR_PREDICTION",
        observed=runtime_evidence["threadpool_before_explicit_context"],
    )
    amendment = validate_amendment(root)
    amendment_v4 = validate_amendment_v4(root, amendment)
    incident = validate_v3_failure_incident(root)
    amendment_v5 = validate_amendment_v5(
        root, amendment, amendment_v4, incident
    )
    bound_paths = revalidate_all_bound_inputs(root, amendment)
    validation = dataclasses.replace(
        validation,
        runtime_identity_pass=True,
    )
    authorization = PredataAuthorization(
        validation,
        artifact_root=root,
        threadpool_recorder=recorder,
        bound_input_identities_rehashed=True,
    )
    validate_pre_root_source_metadata_gate(
        root,
        bound_paths,
        amendment,
        amendment_v5,
        authorization=authorization,
    )
    create_v5_namespace_exclusive(root)
    alias_identities = tuple(
        publish_exact_byte_alias(root, spec, authorization=authorization)
        for spec in ALIAS_SPECS
    )
    if len(alias_identities) != 2:
        raise AssertionError("alias identity count drift")
    authorization.alias_files_published = True
    validate_published_alias_closure(
        root, alias_identities, authorization=authorization
    )
    authorization.authorize_arrow_handoff()
    joined = load_and_join_bound_inputs(
        root, bound_paths, authorization=authorization
    )
    _validate_recomputed_physical_pass(joined)
    return PreparedExecution(
        artifact_root=root,
        attempt_id=attempt_id,
        amendment=amendment,
        amendment_v4=amendment_v4,
        amendment_v5=amendment_v5,
        v3_failure_incident=incident,
        bound_paths=bound_paths,
        authorization=authorization,
        joined=joined,
        runtime_evidence=runtime_evidence,
        alias_identities=alias_identities,
    )


def _value_and_hex(value: Any | None) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    return persist_float(value)


def _stability_record(
    *,
    diagnostic_kind: str,
    diagnostic: str,
    scope_type: str,
    scope_label: str,
    metric: MetricResult,
    weak_floor_applicable: bool,
    strength: CellStrength | None,
) -> dict[str, Any]:
    sst, sst_hex = _value_and_hex(metric.sst)
    sse, sse_hex = _value_and_hex(metric.sse)
    rmse, rmse_hex = _value_and_hex(metric.rmse)
    r2, r2_hex = _value_and_hex(metric.r2)
    nrmse, nrmse_hex = _value_and_hex(metric.nrmse)
    denominator, denominator_hex = persist_float(metric.denominator)
    validate_validity_reason(
        metric.valid, metric.invalid_reason, "stability metric"
    )
    if not metric.valid:
        verdict = "UNUSABLE"
    elif strength is None:
        verdict = "UNRESOLVED"
    elif (
        strength.r2_boundary_equal
        or strength.nrmse_boundary_equal
        or strength.weak_floor_boundary_equal
    ):
        verdict = "AMBIGUOUS_THRESHOLD_EQUALITY"
    elif strength.strict_not_exact_pass and (
        not weak_floor_applicable or bool(strength.weak_floor_pass)
    ):
        verdict = "PASS_STRICT_INTERIOR"
    else:
        verdict = "CLEAR_FAIL_STRICT_INTERIOR"
    return {
        "diagnostic_kind": diagnostic_kind,
        "response_or_endpoint": diagnostic,
        "scope_type": scope_type,
        "scope_label": scope_label,
        "rows": metric.rows,
        "valid": metric.valid,
        "invalid_reason": metric.invalid_reason,
        "unique_count": metric.unique_count,
        "sst": sst,
        "sst_float_hex": sst_hex,
        "sse": sse,
        "sse_float_hex": sse_hex,
        "rmse": rmse,
        "rmse_float_hex": rmse_hex,
        "r2": r2,
        "r2_float_hex": r2_hex,
        "nrmse": nrmse,
        "nrmse_float_hex": nrmse_hex,
        "pooled_nrmse_denominator": denominator,
        "pooled_nrmse_denominator_float_hex": denominator_hex,
        "r2_boundary_equal_0_995": (
            strength.r2_boundary_equal if strength is not None else False
        ),
        "nrmse_boundary_equal_0_02": (
            strength.nrmse_boundary_equal if strength is not None else False
        ),
        "weak_floor_applicable": weak_floor_applicable,
        "nrmse_boundary_equal_0_01": (
            strength.weak_floor_boundary_equal
            if strength is not None and weak_floor_applicable
            else False
        ),
        "strict_not_exact_pass": (
            strength.strict_not_exact_pass if strength is not None else False
        ),
        "weak_floor_pass": (
            strength.weak_floor_pass
            if strength is not None and weak_floor_applicable
            else None
        ),
        "cell_verdict": verdict,
    }


def _aggregate_actual_prediction(
    site: Any, actual: Any, prediction: Any
) -> Any:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "valid_time_utc": site["valid_time_utc"].to_numpy(copy=True),
            "group": site["group__external"].to_numpy(copy=True),
            "capacity_mw": site["capacity_mw__external"].to_numpy(
                dtype=np_float64(0.0).dtype, copy=True
            ),
            "__actual": _float64_vector(actual, "group actual"),
            "__prediction": _float64_vector(prediction, "group prediction"),
        }
    )
    return capacity_aggregate_raw_uv(
        frame, value_columns=("__actual", "__prediction")
    )


@dataclasses.dataclass(frozen=True, slots=True)
class StabilityEvaluation:
    records: tuple[dict[str, Any], ...]
    witnesses: tuple[dict[str, Any], ...]
    gate: StabilityGate


def evaluate_diagnostic_stability(
    site: Any,
    actual: Any,
    prediction: Any,
    *,
    diagnostic: str,
    diagnostic_kind: str,
    pooled_denominator: Any,
) -> StabilityEvaluation:
    """Evaluate the exact 29 cell records and all applicable boundaries."""

    import pandas as pd

    np = _np()
    y = _float64_vector(actual, f"{diagnostic} actual")
    yhat = _float64_vector(prediction, f"{diagnostic} prediction")
    if y.shape != (EXPECTED_SITE_ROWS,) or yhat.shape != y.shape:
        raise IntegrityError("diagnostic site vector row count differs")
    target_days = pd.to_datetime(
        site["target_operating_day_kst__external"], errors="raise"
    )
    years = target_days.dt.year.to_numpy(dtype=np.int16)
    months = target_days.dt.month.to_numpy(dtype=np.int8)
    forecast_hours = site["forecast_hour"].to_numpy(dtype=np.int16, copy=True)
    groups = site["group__external"].astype(str).to_numpy()
    site_ids = site["site_id"].to_numpy(dtype=np.int16, copy=True)
    group_frame = _aggregate_actual_prediction(site, y, yhat)
    group_labels = (
        "kpx_group_1",
        "kpx_group_2",
        "kpx_group_3",
    )
    cell_specs: list[tuple[str, str, Any, Any, bool]] = []
    for year in (2022, 2023):
        mask = years == year
        cell_specs.append(("YEAR", str(year), y[mask], yhat[mask], False))
    season_months = {
        "DJF": (12, 1, 2),
        "MAM": (3, 4, 5),
        "JJA": (6, 7, 8),
        "SON": (9, 10, 11),
    }
    for label, values in season_months.items():
        mask = np.isin(months, np.asarray(values, dtype=np.int8))
        cell_specs.append(("SEASON", label, y[mask], yhat[mask], False))
    bands = {
        "EARLY": (28, 35),
        "MIDDLE": (36, 43),
        "LATE": (44, 51),
    }
    for label, (lower, upper) in bands.items():
        mask = (forecast_hours >= lower) & (forecast_hours <= upper)
        cell_specs.append(
            ("FORECAST_HOUR", label, y[mask], yhat[mask], True)
        )
    for label in group_labels:
        mask = group_frame["group"].to_numpy() == label
        cell_specs.append(
            (
                "GROUP",
                label,
                group_frame.loc[mask, "__actual"].to_numpy(dtype=np.float64),
                group_frame.loc[mask, "__prediction"].to_numpy(dtype=np.float64),
                False,
            )
        )
    observed_sites = tuple(sorted(int(value) for value in np.unique(site_ids)))
    if observed_sites != tuple(range(1, 18)):
        raise IntegrityError("site identifiers differ from exact 01..17")
    for site_id in observed_sites:
        mask = site_ids == site_id
        cell_specs.append(
            ("SITE", f"{site_id:02d}", y[mask], yhat[mask], False)
        )
    if len(cell_specs) != 29:
        raise AssertionError("diagnostic cell slot count drift")

    records: list[dict[str, Any]] = []
    strengths: list[CellStrength | None] = []
    metrics: list[MetricResult] = []
    witnesses: list[dict[str, Any]] = []
    scope_prefix = "RAW" if diagnostic in RESPONSES else "ENDPOINT"
    for scope_type, label, cell_y, cell_prediction, weak in cell_specs:
        metric = regression_metrics(
            cell_y,
            cell_prediction,
            pooled_denominator,
            minimum_rows=MINIMUM_CELL_ROWS,
        )
        strength = (
            evaluate_cell_strength(
                metric,
                weak_floor_applicable=weak,
                stop_on_boundary=False,
            )
            if metric.valid
            else None
        )
        scope = f"{scope_prefix}/{diagnostic}/CELL/{scope_type}/{label}"
        if metric.valid:
            assert metric.r2 is not None and metric.nrmse is not None
            witnesses.extend(
                (
                    make_threshold_witness(
                        cutoff_name="CELL_R2_0.995",
                        scope=scope,
                        diagnostic=diagnostic,
                        value=metric.r2,
                        cutoff=CELL_R2_CUTOFF,
                    ),
                    make_threshold_witness(
                        cutoff_name="CELL_NRMSE_0.02",
                        scope=scope,
                        diagnostic=diagnostic,
                        value=metric.nrmse,
                        cutoff=CELL_NRMSE_CUTOFF,
                    ),
                )
            )
            if weak:
                witnesses.append(
                    make_threshold_witness(
                        cutoff_name="CELL_WEAK_FLOOR_NRMSE_0.01",
                        scope=scope,
                        diagnostic=diagnostic,
                        value=metric.nrmse,
                        cutoff=CELL_WEAK_FLOOR_NRMSE_CUTOFF,
                    )
                )
        metrics.append(metric)
        strengths.append(strength)
        records.append(
            _stability_record(
                diagnostic_kind=diagnostic_kind,
                diagnostic=diagnostic,
                scope_type=scope_type,
                scope_label=label,
                metric=metric,
                weak_floor_applicable=weak,
                strength=strength,
            )
        )

    season_strengths = strengths[2:6]
    season_strict_count = sum(
        item is not None and item.strict_not_exact_pass
        for item in season_strengths
    )
    if season_strict_count == 3:
        remaining_offset = next(
            index
            for index, item in enumerate(season_strengths)
            if item is None or not item.strict_not_exact_pass
        )
        index = 2 + remaining_offset
        metric = metrics[index]
        if metric.valid:
            assert metric.nrmse is not None
            old = strengths[index]
            assert old is not None
            weak_equal = _equal(
                metric.nrmse, CELL_WEAK_FLOOR_NRMSE_CUTOFF
            )
            strengths[index] = dataclasses.replace(
                old,
                weak_floor_pass=bool(
                    np.float64(metric.nrmse)
                    > np.float64(CELL_WEAK_FLOOR_NRMSE_CUTOFF)
                ),
                weak_floor_boundary_equal=weak_equal,
            )
            scope = (
                f"{scope_prefix}/{diagnostic}/CELL/SEASON/"
                f"{cell_specs[index][1]}"
            )
            witnesses.append(
                make_threshold_witness(
                    cutoff_name="CELL_WEAK_FLOOR_NRMSE_0.01",
                    scope=scope,
                    diagnostic=diagnostic,
                    value=metric.nrmse,
                    cutoff=CELL_WEAK_FLOOR_NRMSE_CUTOFF,
                )
            )
            records[index] = _stability_record(
                diagnostic_kind=diagnostic_kind,
                diagnostic=diagnostic,
                scope_type="SEASON",
                scope_label=cell_specs[index][1],
                metric=metric,
                weak_floor_applicable=True,
                strength=strengths[index],
            )
    year_pass = all(
        item is not None
        and item.strict_not_exact_pass
        and not item.r2_boundary_equal
        and not item.nrmse_boundary_equal
        for item in strengths[0:2]
    )
    season_boundary = any(
        item is not None
        and (
            item.r2_boundary_equal
            or item.nrmse_boundary_equal
            or item.weak_floor_boundary_equal
        )
        for item in strengths[2:6]
    )
    if season_strict_count == 4:
        season_pass = not season_boundary
    elif season_strict_count == 3:
        remaining = next(
            item
            for item in strengths[2:6]
            if item is None or not item.strict_not_exact_pass
        )
        season_pass = bool(
            remaining is not None
            and remaining.weak_floor_pass
            and not season_boundary
        )
    else:
        season_pass = False
    forecast_pass = all(
        item is not None
        and item.strict_not_exact_pass
        and bool(item.weak_floor_pass)
        and not item.r2_boundary_equal
        and not item.nrmse_boundary_equal
        and not item.weak_floor_boundary_equal
        for item in strengths[6:9]
    )
    group_pass = all(
        item is not None
        and item.strict_not_exact_pass
        and not item.r2_boundary_equal
        and not item.nrmse_boundary_equal
        for item in strengths[9:12]
    )
    site_groups = {
        f"{int(site_id):02d}": str(group)
        for site_id, group in zip(site_ids, groups, strict=True)
    }
    required = {"kpx_group_1": 5, "kpx_group_2": 5, "kpx_group_3": 4}
    totals = {"kpx_group_1": 0, "kpx_group_2": 0, "kpx_group_3": 0}
    passes = {"kpx_group_1": 0, "kpx_group_2": 0, "kpx_group_3": 0}
    for offset, site_id in enumerate(range(1, 18)):
        group = site_groups[f"{site_id:02d}"]
        totals[group] += 1
        item = strengths[12 + offset]
        passes[group] += int(
            item is not None
            and item.strict_not_exact_pass
            and not item.r2_boundary_equal
            and not item.nrmse_boundary_equal
        )
    if totals != {"kpx_group_1": 6, "kpx_group_2": 6, "kpx_group_3": 5}:
        raise IntegrityError("site group membership count differs")
    site_pass = all(passes[group] >= required[group] for group in required)
    return StabilityEvaluation(
        records=tuple(records),
        witnesses=tuple(witnesses),
        gate=StabilityGate(
            year_pass,
            season_pass,
            forecast_pass,
            group_pass,
            site_pass,
        ),
    )


DISTRIBUTION_RECORD_KEYS: Final[tuple[str, ...]] = (
    "record_kind",
    "diagnostic",
    "scope",
    "month",
    "valid",
    "invalid_reason",
    "iqr_2022",
    "iqr_2022_float_hex",
    "iqr_2023",
    "iqr_2023_float_hex",
    "iqr_ratio",
    "iqr_ratio_float_hex",
    "iqr_boundary_equal_0_1_or_10",
    "reference_year",
    "comparison_year",
    "reference_internal_edges",
    "reference_internal_edges_float_hex",
    "final_bin_count",
    "reference_counts",
    "comparison_counts",
    "reference_probabilities",
    "reference_probabilities_float_hex",
    "comparison_probabilities",
    "comparison_probabilities_float_hex",
    "psi",
    "psi_float_hex",
    "psi_boundary_equal_0_5",
    "verdict",
)


@dataclasses.dataclass(frozen=True, slots=True)
class DistributionEvaluation:
    records: tuple[dict[str, Any], ...]
    witnesses: tuple[dict[str, Any], ...]
    passes: bool


def _float_list(values: Any) -> tuple[list[float], list[str]]:
    decimals: list[float] = []
    witnesses: list[str] = []
    for value in values:
        decimal, witness = persist_float(value)
        decimals.append(decimal)
        witnesses.append(witness)
    return decimals, witnesses


def evaluate_diagnostic_distribution(
    site: Any,
    actual: Any,
    *,
    diagnostic: str,
) -> DistributionEvaluation:
    """Create 52 fixed distribution records using 2022 as PSI reference."""

    import pandas as pd

    np = _np()
    y = _float64_vector(actual, f"{diagnostic} distribution actual")
    if y.shape != (EXPECTED_SITE_ROWS,):
        raise IntegrityError("distribution actual row count differs")
    days = pd.to_datetime(
        site["target_operating_day_kst__external"], errors="raise"
    )
    site_year = days.dt.year.to_numpy(dtype=np.int16)
    site_month = days.dt.month.to_numpy(dtype=np.int8)
    group_actual = _aggregate_actual_prediction(site, y, y)
    valid_to_day = (
        pd.DataFrame(
            {
                "valid_time_utc": site["valid_time_utc"],
                "target_operating_day_kst": site[
                    "target_operating_day_kst__external"
                ],
            }
        )
        .drop_duplicates("valid_time_utc")
        .set_index("valid_time_utc")["target_operating_day_kst"]
    )
    group_days = pd.to_datetime(
        group_actual["valid_time_utc"].map(valid_to_day), errors="raise"
    )
    group_year = group_days.dt.year.to_numpy(dtype=np.int16)
    group_month = group_days.dt.month.to_numpy(dtype=np.int8)
    scope_values: list[tuple[str, Any, Any, Any]] = [
        ("SITE_POOLED", y, site_year, site_month)
    ]
    for group in (
        "kpx_group_1",
        "kpx_group_2",
        "kpx_group_3",
    ):
        mask = group_actual["group"].to_numpy() == group
        scope_values.append(
            (
                group,
                group_actual.loc[mask, "__actual"].to_numpy(dtype=np.float64),
                group_year[mask],
                group_month[mask],
            )
        )
    records: list[dict[str, Any]] = []
    witnesses: list[dict[str, Any]] = []
    passes = True
    prefix = "RAW" if diagnostic in RESPONSES else "ENDPOINT"
    for scope, values, years, months in scope_values:
        for month in range(1, 13):
            y22 = values[(years == 2022) & (months == month)]
            y23 = values[(years == 2023) & (months == month)]
            expected_rows = 816 if scope == "SITE_POOLED" else 48
            if y22.size != expected_rows or y23.size != expected_rows:
                raise IntegrityError("monthly distribution row count differs")
            result = monthly_iqr_stability(
                y22, y23, stop_on_boundary=False
            )
            iqr22, iqr22_hex = persist_float(result.iqr_2022)
            iqr23, iqr23_hex = persist_float(result.iqr_2023)
            ratio, ratio_hex = _value_and_hex(result.ratio)
            if not result.valid:
                verdict = "CLEAR_FAIL_UNUSABLE"
                passes = False
            elif result.boundary_equal:
                verdict = "AMBIGUOUS_THRESHOLD_EQUALITY"
                passes = False
            elif result.passes:
                verdict = "PASS_STRICT_INTERIOR"
            else:
                verdict = "CLEAR_FAIL_STRICT_INTERIOR"
                passes = False
            record = {
                "record_kind": "MONTHLY_IQR",
                "diagnostic": diagnostic,
                "scope": scope,
                "month": month,
                "valid": result.valid,
                "invalid_reason": result.invalid_reason,
                "iqr_2022": iqr22,
                "iqr_2022_float_hex": iqr22_hex,
                "iqr_2023": iqr23,
                "iqr_2023_float_hex": iqr23_hex,
                "iqr_ratio": ratio,
                "iqr_ratio_float_hex": ratio_hex,
                "iqr_boundary_equal_0_1_or_10": result.boundary_equal,
                "reference_year": None,
                "comparison_year": None,
                "reference_internal_edges": [],
                "reference_internal_edges_float_hex": [],
                "final_bin_count": None,
                "reference_counts": [],
                "comparison_counts": [],
                "reference_probabilities": [],
                "reference_probabilities_float_hex": [],
                "comparison_probabilities": [],
                "comparison_probabilities_float_hex": [],
                "psi": None,
                "psi_float_hex": None,
                "psi_boundary_equal_0_5": False,
                "verdict": verdict,
            }
            if tuple(record) != DISTRIBUTION_RECORD_KEYS:
                raise AssertionError("monthly distribution schema drift")
            records.append(record)
            if result.valid and result.ratio is not None:
                threshold_scope = (
                    f"{prefix}/{diagnostic}/DISTRIBUTION/{scope}/"
                    f"MONTH/{month:02d}"
                )
                witnesses.extend(
                    (
                        make_threshold_witness(
                            cutoff_name="MONTHLY_IQR_RATIO_0.1",
                            scope=threshold_scope,
                            diagnostic=diagnostic,
                            value=result.ratio,
                            cutoff=IQR_RATIO_MIN,
                        ),
                        make_threshold_witness(
                            cutoff_name="MONTHLY_IQR_RATIO_10",
                            scope=threshold_scope,
                            diagnostic=diagnostic,
                            value=result.ratio,
                            cutoff=IQR_RATIO_MAX,
                        ),
                    )
                )
        reference = values[years == 2022]
        comparison = values[years == 2023]
        result = year_psi(reference, comparison, stop_on_boundary=False)
        edges, edge_hexes = _float_list(result.internal_edges)
        reference_probabilities, reference_probability_hexes = _float_list(
            result.reference_probabilities
        )
        comparison_probabilities, comparison_probability_hexes = _float_list(
            result.comparison_probabilities
        )
        psi, psi_hex = _value_and_hex(result.psi)
        if not result.valid:
            verdict = "CLEAR_FAIL_UNUSABLE"
            passes = False
        elif result.boundary_equal:
            verdict = "AMBIGUOUS_THRESHOLD_EQUALITY"
            passes = False
        elif result.passes:
            verdict = "PASS_STRICT_INTERIOR"
        else:
            verdict = "CLEAR_FAIL_STRICT_INTERIOR"
            passes = False
        record = {
            "record_kind": "YEAR_PSI",
            "diagnostic": diagnostic,
            "scope": scope,
            "month": None,
            "valid": result.valid,
            "invalid_reason": result.invalid_reason,
            "iqr_2022": None,
            "iqr_2022_float_hex": None,
            "iqr_2023": None,
            "iqr_2023_float_hex": None,
            "iqr_ratio": None,
            "iqr_ratio_float_hex": None,
            "iqr_boundary_equal_0_1_or_10": False,
            "reference_year": 2022,
            "comparison_year": 2023,
            "reference_internal_edges": edges,
            "reference_internal_edges_float_hex": edge_hexes,
            "final_bin_count": result.final_bin_count,
            "reference_counts": [
                int(value) for value in result.reference_counts
            ],
            "comparison_counts": [
                int(value) for value in result.comparison_counts
            ],
            "reference_probabilities": reference_probabilities,
            "reference_probabilities_float_hex": reference_probability_hexes,
            "comparison_probabilities": comparison_probabilities,
            "comparison_probabilities_float_hex": comparison_probability_hexes,
            "psi": psi,
            "psi_float_hex": psi_hex,
            "psi_boundary_equal_0_5": result.boundary_equal,
            "verdict": verdict,
        }
        if tuple(record) != DISTRIBUTION_RECORD_KEYS:
            raise AssertionError("PSI distribution schema drift")
        records.append(record)
        if result.valid and result.psi is not None:
            witnesses.append(
                make_threshold_witness(
                    cutoff_name="YEAR_PSI_0.5",
                    scope=(
                        f"{prefix}/{diagnostic}/DISTRIBUTION/{scope}/"
                        "YEAR/2022_2023"
                    ),
                    diagnostic=diagnostic,
                    value=result.psi,
                    cutoff=PSI_MAX,
                )
            )
    if len(records) != 52:
        raise AssertionError("per-diagnostic distribution record count drift")
    return DistributionEvaluation(tuple(records), tuple(witnesses), passes)


def _empty_metric_result(
    rows: int, denominator: Any, reason: str
) -> MetricResult:
    return MetricResult(
        rows=rows,
        valid=False,
        invalid_reason=reason,
        unique_count=0,
        sst=None,
        sse=None,
        rmse=None,
        r2=None,
        nrmse=None,
        denominator=_finite_float64(denominator, "empty metric denominator"),
    )


def empty_stability_records(
    diagnostic: str,
    *,
    diagnostic_kind: str,
    denominator: Any,
    reason: str,
) -> tuple[dict[str, Any], ...]:
    specs = (
        *[("YEAR", value, 9_792) for value in ("2022", "2023")],
        *[
            ("SEASON", value, 4_896)
            for value in ("DJF", "MAM", "JJA", "SON")
        ],
        *[
            ("FORECAST_HOUR", value, 6_528)
            for value in ("EARLY", "MIDDLE", "LATE")
        ],
        *[
            ("GROUP", value, 1_152)
            for value in (
                "kpx_group_1",
                "kpx_group_2",
                "kpx_group_3",
            )
        ],
        *[("SITE", f"{value:02d}", 1_152) for value in range(1, 18)],
    )
    records = tuple(
        _stability_record(
            diagnostic_kind=diagnostic_kind,
            diagnostic=diagnostic,
            scope_type=scope,
            scope_label=label,
            metric=_empty_metric_result(rows, denominator, reason),
            weak_floor_applicable=False,
            strength=None,
        )
        for scope, label, rows in specs
    )
    if len(records) != 29:
        raise AssertionError("empty stability record count drift")
    return records


def empty_distribution_records(
    diagnostic: str, *, reason: str
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for scope in (
        "SITE_POOLED",
        "kpx_group_1",
        "kpx_group_2",
        "kpx_group_3",
    ):
        for month in range(1, 13):
            records.append(
                {
                    "record_kind": "MONTHLY_IQR",
                    "diagnostic": diagnostic,
                    "scope": scope,
                    "month": month,
                    "valid": False,
                    "invalid_reason": reason,
                    "iqr_2022": None,
                    "iqr_2022_float_hex": None,
                    "iqr_2023": None,
                    "iqr_2023_float_hex": None,
                    "iqr_ratio": None,
                    "iqr_ratio_float_hex": None,
                    "iqr_boundary_equal_0_1_or_10": False,
                    "reference_year": None,
                    "comparison_year": None,
                    "reference_internal_edges": [],
                    "reference_internal_edges_float_hex": [],
                    "final_bin_count": None,
                    "reference_counts": [],
                    "comparison_counts": [],
                    "reference_probabilities": [],
                    "reference_probabilities_float_hex": [],
                    "comparison_probabilities": [],
                    "comparison_probabilities_float_hex": [],
                    "psi": None,
                    "psi_float_hex": None,
                    "psi_boundary_equal_0_5": False,
                    "verdict": "STRUCTURALLY_NOT_APPLICABLE",
                }
            )
        records.append(
            {
                "record_kind": "YEAR_PSI",
                "diagnostic": diagnostic,
                "scope": scope,
                "month": None,
                "valid": False,
                "invalid_reason": reason,
                "iqr_2022": None,
                "iqr_2022_float_hex": None,
                "iqr_2023": None,
                "iqr_2023_float_hex": None,
                "iqr_ratio": None,
                "iqr_ratio_float_hex": None,
                "iqr_boundary_equal_0_1_or_10": False,
                "reference_year": 2022,
                "comparison_year": 2023,
                "reference_internal_edges": [],
                "reference_internal_edges_float_hex": [],
                "final_bin_count": None,
                "reference_counts": [],
                "comparison_counts": [],
                "reference_probabilities": [],
                "reference_probabilities_float_hex": [],
                "comparison_probabilities": [],
                "comparison_probabilities_float_hex": [],
                "psi": None,
                "psi_float_hex": None,
                "psi_boundary_equal_0_5": False,
                "verdict": "STRUCTURALLY_NOT_APPLICABLE",
            }
        )
    if len(records) != 52 or any(
        tuple(record) != DISTRIBUTION_RECORD_KEYS for record in records
    ):
        raise AssertionError("empty distribution record schema drift")
    return tuple(records)


def _duplicate_metric_record(
    *,
    record_kind: str,
    response: str,
    estimator: str | None,
    predictor: str | None,
    metric: MetricResult,
    pooled: PooledDenominator,
    pearson: Any | None = None,
    spearman: Any | None = None,
    affine_oof_nrmse: Any | None = None,
    independent_selected: bool = False,
    parent_max_r2_witness: bool = False,
    parent_redundancy_boolean: bool | None = None,
    affine_qualifies: bool = False,
    boundary_equality_flags: Sequence[str] = (),
    component_classification: str = "",
) -> dict[str, Any]:
    validate_validity_reason(
        metric.valid, metric.invalid_reason, "duplicate pooled metric"
    )
    sse, sse_hex = _value_and_hex(metric.sse)
    rmse, rmse_hex = _value_and_hex(metric.rmse)
    r2, r2_hex = _value_and_hex(metric.r2)
    nrmse, nrmse_hex = _value_and_hex(metric.nrmse)
    q05, q05_hex = persist_float(pooled.q05)
    q95, q95_hex = persist_float(pooled.q95)
    denominator, denominator_hex = persist_float(pooled.denominator)
    pearson_value, pearson_hex = _value_and_hex(pearson)
    spearman_value, spearman_hex = _value_and_hex(spearman)
    affine_value, affine_hex = _value_and_hex(affine_oof_nrmse)
    return {
        "record_kind": record_kind,
        "response": response,
        "estimator": estimator,
        "predictor": predictor,
        "rows": metric.rows,
        "valid": metric.valid,
        "invalid_reason": metric.invalid_reason,
        "sse": sse,
        "sse_float_hex": sse_hex,
        "rmse": rmse,
        "rmse_float_hex": rmse_hex,
        "r2": r2,
        "r2_float_hex": r2_hex,
        "nrmse": nrmse,
        "nrmse_float_hex": nrmse_hex,
        "actual_q05": q05,
        "actual_q05_float_hex": q05_hex,
        "actual_q95": q95,
        "actual_q95_float_hex": q95_hex,
        "nrmse_denominator": denominator,
        "nrmse_denominator_float_hex": denominator_hex,
        "pearson": pearson_value,
        "pearson_float_hex": pearson_hex,
        "spearman": spearman_value,
        "spearman_float_hex": spearman_hex,
        "affine_oof_nrmse": affine_value,
        "affine_oof_nrmse_float_hex": affine_hex,
        "independent_selected": independent_selected,
        "parent_max_r2_witness": parent_max_r2_witness,
        "parent_redundancy_boolean": parent_redundancy_boolean,
        "affine_qualifies": affine_qualifies,
        "boundary_equality_flags": ";".join(boundary_equality_flags),
        "component_classification": component_classification,
    }


@dataclasses.dataclass(frozen=True, slots=True)
class ResponseEvaluation:
    response: str
    actual: Any
    pooled: PooledDenominator
    ridge_oof: Any | None
    extra_trees_oof: Any | None
    independent_oof: Any | None
    independent_estimator: str | None
    raw_result: Mapping[str, Any]
    duplicate_metric_records: tuple[dict[str, Any], ...]
    stability_records: tuple[dict[str, Any], ...]
    distribution_records: tuple[dict[str, Any], ...]
    threshold_witnesses: tuple[dict[str, Any], ...]
    tied_witnesses: tuple[dict[str, Any], ...]
    completed_fit_records: tuple[dict[str, Any], ...]
    skip_reason: str | None


def _selection_without_boundary_stop(
    ridge: EstimatorMetric, trees: EstimatorMetric
) -> tuple[
    EstimatorMetric | None,
    tuple[EstimatorMetric, ...],
    tuple[bool, ...],
    bool,
]:
    np = _np()
    ridge.validated()
    trees.validated()
    rmse_tie = bool(np.float64(ridge.rmse) == np.float64(trees.rmse))
    independent = (
        None
        if rmse_tie
        else ridge
        if bool(np.float64(ridge.rmse) < np.float64(trees.rmse))
        else trees
    )
    maximum = np.float64(max(np.float64(ridge.r2), np.float64(trees.r2)))
    parents = tuple(
        item
        for item in (ridge, trees)
        if bool(np.float64(item.r2) == maximum)
    )
    parent_booleans = tuple(
        bool(
            np.float64(item.r2) > np.float64(PARENT_R2_CUTOFF)
            and np.float64(item.nrmse) < np.float64(PARENT_NRMSE_CUTOFF)
        )
        for item in parents
    )
    return independent, parents, parent_booleans, rmse_tie


def evaluate_raw_response(
    prepared: PreparedExecution, response: str
) -> ResponseEvaluation:
    """Fit/evaluate one response and preserve every fixed diagnostic witness."""

    np = _np()
    if response not in RESPONSES:
        raise IntegrityError(f"unknown response: {response}")
    site = prepared.joined.site
    y = _float64_vector(site[response].to_numpy(), f"{response} actual")
    pooled = pooled_nrmse_denominator(y)
    skip_reason = premodel_response_skip_reason(y)
    if skip_reason is not None:
        classification = "CLEAR_NONNOVEL_CONSTANT"
        invalid = _empty_metric_result(
            EXPECTED_SITE_ROWS, pooled.denominator, skip_reason
        )
        duplicate_records = [
            _duplicate_metric_record(
                record_kind="POOLED_ESTIMATOR",
                response=response,
                estimator=estimator_name,
                predictor=None,
                metric=invalid,
                pooled=pooled,
                component_classification=classification,
            )
            for estimator_name in ("RIDGE_PIPELINE", "EXTRA_TREES")
        ]
        duplicate_records.extend(
            _duplicate_metric_record(
                record_kind="AFFINE_PREDICTOR",
                response=response,
                estimator=None,
                predictor=predictor,
                metric=invalid,
                pooled=pooled,
                component_classification=classification,
            )
            for predictor in PREDICTORS
        )
        return ResponseEvaluation(
            response=response,
            actual=y,
            pooled=pooled,
            ridge_oof=None,
            extra_trees_oof=None,
            independent_oof=None,
            independent_estimator=None,
            raw_result={
                "response": response,
                "classification": classification,
                "counts_as_nonredundant": False,
                "independent_estimator": None,
                "parent_reference_estimators": [],
                "parent_redundancy_boolean": None,
                "affine_qualifying_predictors": [],
                "cell_stability_pass": None,
                "distribution_stability_pass": None,
                "boundary_equality_flags": [],
                "ambiguity_reason": None,
            },
            duplicate_metric_records=tuple(duplicate_records),
            stability_records=empty_stability_records(
                response,
                diagnostic_kind="RAW_RESPONSE",
                denominator=pooled.denominator,
                reason=skip_reason,
            ),
            distribution_records=empty_distribution_records(
                response, reason=skip_reason
            ),
            threshold_witnesses=(),
            tied_witnesses=(),
            completed_fit_records=(),
            skip_reason=skip_reason,
        )

    primary = crossfit_primary_response(
        prepared.joined.predictor_matrix_float64,
        y,
        prepared.joined.fold_ordinals,
        response_name=response,
        authorization=prepared.authorization,
    )
    affine = crossfit_affine_response(
        prepared.joined.predictor_matrix_float64,
        y,
        prepared.joined.fold_ordinals,
        response_name=response,
        pooled_denominator=pooled.denominator,
        authorization=prepared.authorization,
    )
    ridge_metric_result = regression_metrics(
        y, primary.ridge_oof, pooled.denominator
    )
    trees_metric_result = regression_metrics(
        y, primary.extra_trees_oof, pooled.denominator
    )
    ridge_metric_result.require_valid()
    trees_metric_result.require_valid()
    assert (
        ridge_metric_result.rmse is not None
        and ridge_metric_result.r2 is not None
        and ridge_metric_result.nrmse is not None
        and trees_metric_result.rmse is not None
        and trees_metric_result.r2 is not None
        and trees_metric_result.nrmse is not None
    )
    ridge = EstimatorMetric(
        "RIDGE_PIPELINE",
        ridge_metric_result.rmse,
        ridge_metric_result.r2,
        ridge_metric_result.nrmse,
    )
    trees = EstimatorMetric(
        "EXTRA_TREES",
        trees_metric_result.rmse,
        trees_metric_result.r2,
        trees_metric_result.nrmse,
    )
    independent, parents, parent_booleans, rmse_tie = (
        _selection_without_boundary_stop(ridge, trees)
    )
    tied: list[dict[str, Any]] = []
    if rmse_tie:
        tied.append(
            {
                "tie_kind": "INDEPENDENT_RMSE",
                "diagnostic": response,
                "estimators": ["RIDGE_PIPELINE", "EXTRA_TREES"],
                "metric_name": "RMSE",
                "metric_values_float_hex": [
                    canonical_float_hex(ridge.rmse),
                    canonical_float_hex(trees.rmse),
                ],
                "resolution": "GLOBAL_AMBIGUITY",
                "global_ambiguity": True,
            }
        )
        classification = "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
        raw_result = {
            "response": response,
            "classification": classification,
            "counts_as_nonredundant": False,
            "independent_estimator": None,
            "parent_reference_estimators": [],
            "parent_redundancy_boolean": None,
            "affine_qualifying_predictors": [],
            "cell_stability_pass": None,
            "distribution_stability_pass": None,
            "boundary_equality_flags": [],
            "ambiguity_reason": classification,
        }
        stability_records = empty_stability_records(
            response,
            diagnostic_kind="RAW_RESPONSE",
            denominator=pooled.denominator,
            reason="STRUCTURALLY_NOT_APPLICABLE",
        )
        distribution_records = evaluate_diagnostic_distribution(
            site, y, diagnostic=response
        ).records
        witnesses: tuple[dict[str, Any], ...] = ()
        independent_oof = None
        affine_hits: tuple[str, ...] = ()
        parent_boolean: bool | None = None
        stability_pass: bool | None = None
        distribution_pass: bool | None = None
    else:
        assert independent is not None
        independent_oof = (
            primary.ridge_oof
            if independent.estimator == "RIDGE_PIPELINE"
            else primary.extra_trees_oof
        )
        stability = evaluate_diagnostic_stability(
            site,
            y,
            independent_oof,
            diagnostic=response,
            diagnostic_kind="RAW_RESPONSE",
            pooled_denominator=pooled.denominator,
        )
        distribution = evaluate_diagnostic_distribution(
            site, y, diagnostic=response
        )
        witness_list: list[dict[str, Any]] = []
        affine_hits_list: list[str] = []
        affine_by_predictor = {
            screen.predictor: screen for screen in affine.screens
        }
        for predictor in PREDICTORS:
            screen = affine_by_predictor[predictor]
            scope = f"RAW/{response}/AFFINE/{predictor}"
            if screen.pearson is not None:
                witness_list.append(
                    make_threshold_witness(
                        cutoff_name="AFFINE_ABS_PEARSON_0.9999",
                        scope=scope,
                        diagnostic=response,
                        value=np.float64(abs(np.float64(screen.pearson))),
                        cutoff=AFFINE_CORRELATION_CUTOFF,
                    )
                )
            if screen.spearman is not None:
                witness_list.append(
                    make_threshold_witness(
                        cutoff_name="AFFINE_ABS_SPEARMAN_0.9999",
                        scope=scope,
                        diagnostic=response,
                        value=np.float64(abs(np.float64(screen.spearman))),
                        cutoff=AFFINE_CORRELATION_CUTOFF,
                    )
                )
            if screen.affine_oof_nrmse is not None:
                witness_list.append(
                    make_threshold_witness(
                        cutoff_name="AFFINE_NRMSE_0.01",
                        scope=scope,
                        diagnostic=response,
                        value=screen.affine_oof_nrmse,
                        cutoff=AFFINE_NRMSE_CUTOFF,
                    )
                )
            if (
                screen.valid
                and screen.pearson is not None
                and screen.spearman is not None
                and screen.affine_oof_nrmse is not None
                and bool(
                    abs(np.float64(screen.pearson))
                    > np.float64(AFFINE_CORRELATION_CUTOFF)
                    and abs(np.float64(screen.spearman))
                    > np.float64(AFFINE_CORRELATION_CUTOFF)
                    and np.float64(screen.affine_oof_nrmse)
                    < np.float64(AFFINE_NRMSE_CUTOFF)
                )
            ):
                affine_hits_list.append(predictor)
        for parent in parents:
            scope = f"RAW/{response}/PARENT/{parent.estimator}"
            witness_list.extend(
                (
                    make_threshold_witness(
                        cutoff_name="PARENT_R2_0.98",
                        scope=scope,
                        diagnostic=response,
                        value=parent.r2,
                        cutoff=PARENT_R2_CUTOFF,
                    ),
                    make_threshold_witness(
                        cutoff_name="PARENT_NRMSE_0.10",
                        scope=scope,
                        diagnostic=response,
                        value=parent.nrmse,
                        cutoff=PARENT_NRMSE_CUTOFF,
                    ),
                )
            )
        independent_scope = (
            f"RAW/{response}/INDEPENDENT/{independent.estimator}"
        )
        for cutoff_name, value, cutoff in (
            (
                "INDEPENDENT_R2_EXACT_DUPLICATE_0.995",
                independent.r2,
                INDEPENDENT_DUPLICATE_R2_CUTOFF,
            ),
            (
                "INDEPENDENT_NRMSE_EXACT_DUPLICATE_0.02",
                independent.nrmse,
                INDEPENDENT_DUPLICATE_NRMSE_CUTOFF,
            ),
            (
                "INDEPENDENT_R2_STABLE_MARGIN_0.9925",
                independent.r2,
                INDEPENDENT_MARGIN_R2_CUTOFF,
            ),
            (
                "INDEPENDENT_NRMSE_STABLE_MARGIN_0.025",
                independent.nrmse,
                INDEPENDENT_MARGIN_NRMSE_CUTOFF,
            ),
        ):
            witness_list.append(
                make_threshold_witness(
                    cutoff_name=cutoff_name,
                    scope=independent_scope,
                    diagnostic=response,
                    value=value,
                    cutoff=cutoff,
                )
            )
        witness_list.extend(stability.witnesses)
        witness_list.extend(distribution.witnesses)
        witness_list.sort(key=_threshold_sort_key)
        boundary_flags = [
            cutoff
            for cutoff in CUTOFF_NAMES
            if any(
                witness["cutoff_name"] == cutoff
                and witness["boundary_equal"]
                for witness in witness_list
            )
        ]
        affine_hits = tuple(affine_hits_list)
        differing_parent_tie = len(set(parent_booleans)) > 1
        if boundary_flags:
            classification = "AMBIGUOUS_THRESHOLD_EQUALITY"
        elif affine_hits:
            classification = "CLEAR_REDUNDANT_AFFINE"
        elif differing_parent_tie:
            classification = "AMBIGUOUS_PARENT_ESTIMATOR_TIE"
        elif parent_booleans[0]:
            classification = "CLEAR_REDUNDANT_PARENT_OOF"
        elif bool(
            np.float64(independent.r2)
            > np.float64(INDEPENDENT_DUPLICATE_R2_CUTOFF)
            and np.float64(independent.nrmse)
            < np.float64(INDEPENDENT_DUPLICATE_NRMSE_CUTOFF)
        ):
            classification = "CLEAR_REDUNDANT_INDEPENDENT_EXACT_DUPLICATE"
        else:
            margin = bool(
                np.float64(independent.r2)
                < np.float64(INDEPENDENT_MARGIN_R2_CUTOFF)
                and np.float64(independent.nrmse)
                > np.float64(INDEPENDENT_MARGIN_NRMSE_CUTOFF)
            )
            if margin and stability.gate.all_pass and distribution.passes:
                classification = "PASS_NONREDUNDANT_MARGIN"
            elif margin:
                classification = "CLEAR_UNSTABLE_FAILURE"
            else:
                classification = "AMBIGUOUS_GRAY_ZONE"
        if len(parents) == 2 and not boundary_flags:
            if differing_parent_tie:
                if affine_hits and not boundary_flags:
                    resolution = "MASKED_BY_PRIOR_AFFINE_REDUNDANCY"
                    tie_global = False
                else:
                    resolution = "GLOBAL_AMBIGUITY"
                    tie_global = True
            else:
                resolution = (
                    "SHARED_PARENT_REDUNDANCY_TRUE"
                    if parent_booleans[0]
                    else "SHARED_PARENT_REDUNDANCY_FALSE"
                )
                tie_global = False
            tied.append(
                {
                    "tie_kind": "PARENT_MAX_R2",
                    "diagnostic": response,
                    "estimators": ["RIDGE_PIPELINE", "EXTRA_TREES"],
                    "metric_name": "R2",
                    "metric_values_float_hex": [
                        canonical_float_hex(ridge.r2),
                        canonical_float_hex(trees.r2),
                    ],
                    "resolution": resolution,
                    "global_ambiguity": tie_global,
                }
            )
        ambiguous = classification in AMBIGUOUS_CLASSIFICATIONS
        if (
            classification == "CLEAR_REDUNDANT_AFFINE"
            and differing_parent_tie
        ):
            parent_boolean = None
        elif ambiguous:
            parent_boolean = None
        else:
            parent_boolean = parent_booleans[0]
        stability_pass = None if ambiguous else stability.gate.all_pass
        distribution_pass = None if ambiguous else distribution.passes
        raw_result = {
            "response": response,
            "classification": classification,
            "counts_as_nonredundant": (
                classification == "PASS_NONREDUNDANT_MARGIN"
            ),
            "independent_estimator": independent.estimator,
            "parent_reference_estimators": [
                parent.estimator for parent in parents
            ],
            "parent_redundancy_boolean": parent_boolean,
            "affine_qualifying_predictors": list(affine_hits),
            "cell_stability_pass": stability_pass,
            "distribution_stability_pass": distribution_pass,
            "boundary_equality_flags": boundary_flags,
            "ambiguity_reason": classification if ambiguous else None,
        }
        stability_records = stability.records
        distribution_records = distribution.records
        witnesses = tuple(witness_list)

    metric_records: list[dict[str, Any]] = []
    independent_name = (
        independent.estimator if independent is not None else None
    )
    parent_names = {parent.estimator for parent in parents} if not rmse_tie else set()
    parent_bool_map = (
        {}
        if rmse_tie
        else {
            parent.estimator: boolean
            for parent, boolean in zip(parents, parent_booleans, strict=True)
        }
    )
    flags = raw_result["boundary_equality_flags"]
    for estimator_name, metric_result in (
        ("RIDGE_PIPELINE", ridge_metric_result),
        ("EXTRA_TREES", trees_metric_result),
    ):
        metric_records.append(
            _duplicate_metric_record(
                record_kind="POOLED_ESTIMATOR",
                response=response,
                estimator=estimator_name,
                predictor=None,
                metric=metric_result,
                pooled=pooled,
                independent_selected=estimator_name == independent_name,
                parent_max_r2_witness=estimator_name in parent_names,
                parent_redundancy_boolean=parent_bool_map.get(estimator_name),
                boundary_equality_flags=flags,
                component_classification=raw_result["classification"],
            )
        )
    affine_screen_map = {screen.predictor: screen for screen in affine.screens}
    for predictor in PREDICTORS:
        screen = affine_screen_map[predictor]
        affine_metric = regression_metrics(
            y, affine.predictor_oof[predictor], pooled.denominator
        )
        metric_records.append(
            _duplicate_metric_record(
                record_kind="AFFINE_PREDICTOR",
                response=response,
                estimator=None,
                predictor=predictor,
                metric=affine_metric,
                pooled=pooled,
                pearson=screen.pearson,
                spearman=screen.spearman,
                affine_oof_nrmse=screen.affine_oof_nrmse,
                affine_qualifies=predictor in affine_hits,
                boundary_equality_flags=flags,
                component_classification=raw_result["classification"],
            )
        )
    if len(metric_records) != 37:
        raise AssertionError("per-response duplicate metric count drift")
    completed_fit_records = (
        *primary.completed_ledger_records,
        *affine.completed_ledger_records,
    )
    return ResponseEvaluation(
        response=response,
        actual=y,
        pooled=pooled,
        ridge_oof=primary.ridge_oof,
        extra_trees_oof=primary.extra_trees_oof,
        independent_oof=independent_oof,
        independent_estimator=independent_name,
        raw_result=raw_result,
        duplicate_metric_records=tuple(metric_records),
        stability_records=stability_records,
        distribution_records=distribution_records,
        threshold_witnesses=witnesses,
        tied_witnesses=tuple(tied),
        completed_fit_records=tuple(completed_fit_records),
        skip_reason=None,
    )


ENDPOINT_POOLED_KEYS: Final[tuple[str, ...]] = (
    "diagnostic",
    "rows",
    "valid",
    "invalid_reason",
    "source_component_estimator_binding",
    "sse",
    "sse_float_hex",
    "rmse",
    "rmse_float_hex",
    "r2",
    "r2_float_hex",
    "nrmse",
    "nrmse_float_hex",
    "actual_q05",
    "actual_q05_float_hex",
    "actual_q95",
    "actual_q95_float_hex",
    "nrmse_denominator",
    "nrmse_denominator_float_hex",
    "r2_boundary_equal_0_9925",
    "nrmse_boundary_equal_0_025",
    "strict_stable_margin_pass",
    "endpoint_pooled_verdict",
)


def _derived_dependencies(diagnostic: str) -> tuple[str, ...]:
    if diagnostic.startswith("WS_"):
        level = diagnostic.removeprefix("WS_")
        return (f"UGRD_{level}mb", f"VGRD_{level}mb")
    if diagnostic == "ENDPOINT_DU_925_MINUS_1000":
        return ("UGRD_925mb", "UGRD_1000mb")
    if diagnostic == "ENDPOINT_DV_925_MINUS_1000":
        return ("VGRD_925mb", "VGRD_1000mb")
    if diagnostic in {
        "ENDPOINT_VECTOR_SHEAR_MAG",
        "ENDPOINT_DIRECTION_COS",
        "ENDPOINT_DIRECTION_SIN",
    }:
        return (
            "UGRD_925mb",
            "VGRD_925mb",
            "UGRD_1000mb",
            "VGRD_1000mb",
        )
    match = re.fullmatch(
        r"ADJACENT_(DU|DV|SHEAR_MAG)_(925|950|975)_MINUS_(950|975|1000)",
        diagnostic,
    )
    if match is None:
        raise IntegrityError(f"unknown derived diagnostic: {diagnostic}")
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


def derive_partial_wind_diagnostics(
    components: Mapping[str, Any | None],
) -> dict[str, Any | None]:
    """Derive every diagnostic whose exact raw dependency set is available."""

    np = _np()
    result: dict[str, Any | None] = {}
    for diagnostic in DERIVED_DIAGNOSTICS:
        dependencies = _derived_dependencies(diagnostic)
        if any(components.get(name) is None for name in dependencies):
            result[diagnostic] = None
            continue
        values = {
            name: _float64_vector(components[name], f"{diagnostic} source")
            for name in dependencies
        }
        if diagnostic.startswith("WS_"):
            level = diagnostic.removeprefix("WS_")
            output = np.hypot(
                values[f"UGRD_{level}mb"], values[f"VGRD_{level}mb"]
            )
        elif diagnostic == "ENDPOINT_DU_925_MINUS_1000":
            output = values["UGRD_925mb"] - values["UGRD_1000mb"]
        elif diagnostic == "ENDPOINT_DV_925_MINUS_1000":
            output = values["VGRD_925mb"] - values["VGRD_1000mb"]
        elif diagnostic == "ENDPOINT_VECTOR_SHEAR_MAG":
            du = values["UGRD_925mb"] - values["UGRD_1000mb"]
            dv = values["VGRD_925mb"] - values["VGRD_1000mb"]
            output = np.hypot(du, dv)
        elif diagnostic.startswith("ADJACENT_"):
            match = re.fullmatch(
                r"ADJACENT_(DU|DV|SHEAR_MAG)_(925|950|975)_MINUS_(950|975|1000)",
                diagnostic,
            )
            assert match is not None
            kind, upper, lower = match.groups()
            if kind == "DU":
                output = (
                    values[f"UGRD_{upper}mb"] - values[f"UGRD_{lower}mb"]
                )
            elif kind == "DV":
                output = (
                    values[f"VGRD_{upper}mb"] - values[f"VGRD_{lower}mb"]
                )
            else:
                du = (
                    values[f"UGRD_{upper}mb"] - values[f"UGRD_{lower}mb"]
                )
                dv = (
                    values[f"VGRD_{upper}mb"] - values[f"VGRD_{lower}mb"]
                )
                output = np.hypot(du, dv)
        else:
            u1000 = values["UGRD_1000mb"]
            v1000 = values["VGRD_1000mb"]
            u925 = values["UGRD_925mb"]
            v925 = values["VGRD_925mb"]
            ws1000 = np.hypot(u1000, v1000)
            ws925 = np.hypot(u925, v925)
            denominator = np.maximum(
                ws1000 * ws925, np.float64(CALM_THRESHOLD_MPS)
            )
            calm = (ws1000 < np.float64(CALM_THRESHOLD_MPS)) | (
                ws925 < np.float64(CALM_THRESHOLD_MPS)
            )
            if diagnostic == "ENDPOINT_DIRECTION_COS":
                output = np.clip(
                    (u1000 * u925 + v1000 * v925) / denominator, -1.0, 1.0
                )
            elif diagnostic == "ENDPOINT_DIRECTION_SIN":
                output = np.clip(
                    (u1000 * v925 - v1000 * u925) / denominator, -1.0, 1.0
                )
            else:
                raise AssertionError("derived diagnostic branch drift")
            output = np.ascontiguousarray(output, dtype=np.float64)
            output[calm] = np.float64(0.0)
        output = np.ascontiguousarray(output, dtype=np.float64)
        if not bool(np.isfinite(output).all()):
            raise IntegrityError(f"derived diagnostic is non-finite: {diagnostic}")
        result[diagnostic] = output
    return result


def _capacity_aggregate_component_mapping(
    site: Any, components: Mapping[str, Any | None]
) -> tuple[Any, dict[str, Any | None]]:
    import pandas as pd

    np = _np()
    available = {
        name: value for name, value in components.items() if value is not None
    }
    frame = pd.DataFrame(
        {
            "valid_time_utc": site["valid_time_utc"].to_numpy(copy=True),
            "group": site["group__external"].to_numpy(copy=True),
            "capacity_mw": site["capacity_mw__external"].to_numpy(
                dtype=np.float64, copy=True
            ),
            **{
                name: _float64_vector(value, f"group source {name}")
                for name, value in available.items()
            },
        }
    )
    if available:
        aggregated = capacity_aggregate_raw_uv(
            frame, value_columns=tuple(available)
        )
    else:
        aggregated = frame.loc[
            :, ["valid_time_utc", "group", "capacity_mw"]
        ].drop_duplicates(["valid_time_utc", "group"])
        aggregated = aggregated.sort_values(
            ["valid_time_utc", "group"], kind="mergesort", ignore_index=True
        )
    result = {
        name: (
            aggregated[name].to_numpy(dtype=np.float64, copy=True)
            if name in available
            else None
        )
        for name in components
    }
    return aggregated, result


@dataclasses.dataclass(frozen=True, slots=True)
class DerivedEvaluation:
    site_actual: Mapping[str, Any]
    site_prediction: Mapping[str, Any | None]
    group_metadata: Any
    group_actual: Mapping[str, Any]
    group_prediction: Mapping[str, Any | None]
    endpoint_results: tuple[dict[str, Any], ...]
    endpoint_pooled_records: tuple[dict[str, Any], ...]
    stability_records: tuple[dict[str, Any], ...]
    distribution_records: tuple[dict[str, Any], ...]
    threshold_witnesses: tuple[dict[str, Any], ...]


def evaluate_derived_wind(
    prepared: PreparedExecution,
    raw_evaluations: Mapping[str, ResponseEvaluation],
) -> DerivedEvaluation:
    np = _np()
    site = prepared.joined.site
    raw_actual = {
        response: raw_evaluations[response].actual
        for response in WIND_RESPONSES
    }
    raw_prediction = {
        response: raw_evaluations[response].independent_oof
        for response in WIND_RESPONSES
    }
    site_actual = derive_partial_wind_diagnostics(raw_actual)
    site_prediction = derive_partial_wind_diagnostics(raw_prediction)
    if any(value is None for value in site_actual.values()):
        raise IntegrityError("finite raw actual failed complete derived construction")
    group_metadata, group_raw_actual = _capacity_aggregate_component_mapping(
        site, raw_actual
    )
    _, group_raw_prediction = _capacity_aggregate_component_mapping(
        site, raw_prediction
    )
    group_actual = derive_partial_wind_diagnostics(group_raw_actual)
    group_prediction = derive_partial_wind_diagnostics(group_raw_prediction)
    if any(value is None for value in group_actual.values()):
        raise IntegrityError("group actual failed complete derived construction")

    endpoint_results: list[dict[str, Any]] = []
    pooled_records: list[dict[str, Any]] = []
    stability_records: list[dict[str, Any]] = []
    distribution_records: list[dict[str, Any]] = []
    witnesses: list[dict[str, Any]] = []
    for diagnostic in ENDPOINT_VETO_DIAGNOSTICS:
        source_responses = _endpoint_source_responses(diagnostic)
        binding = {
            response: raw_evaluations[response].independent_estimator
            for response in source_responses
        }
        source_tie = any(
            raw_evaluations[response].raw_result["classification"]
            == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
            for response in source_responses
        )
        source_unavailable = any(value is None for value in binding.values())
        actual = site_actual[diagnostic]
        assert actual is not None
        pooled = pooled_nrmse_denominator(actual)
        if source_unavailable:
            verdict = (
                "AMBIGUOUS_RAW_ESTIMATOR_TIE"
                if source_tie
                else "CLEAR_ENDPOINT_VETO_FAILURE"
            )
            # The endpoint schema has no raw-selector-tie invalid-reason enum.
            # The endpoint verdict and raw/global records preserve that cause;
            # unavailable endpoint metric evidence is structurally inapplicable.
            reason = "STRUCTURALLY_NOT_APPLICABLE"
            invalid = _empty_metric_result(
                EXPECTED_SITE_ROWS, pooled.denominator, reason
            )
            sse = sse_hex = rmse = rmse_hex = r2 = r2_hex = None
            nrmse = nrmse_hex = None
            q05, q05_hex = persist_float(pooled.q05)
            q95, q95_hex = persist_float(pooled.q95)
            denominator, denominator_hex = persist_float(pooled.denominator)
            pooled_record = {
                "diagnostic": diagnostic,
                "rows": EXPECTED_SITE_ROWS,
                "valid": False,
                "invalid_reason": reason,
                "source_component_estimator_binding": binding,
                "sse": sse,
                "sse_float_hex": sse_hex,
                "rmse": rmse,
                "rmse_float_hex": rmse_hex,
                "r2": r2,
                "r2_float_hex": r2_hex,
                "nrmse": nrmse,
                "nrmse_float_hex": nrmse_hex,
                "actual_q05": q05,
                "actual_q05_float_hex": q05_hex,
                "actual_q95": q95,
                "actual_q95_float_hex": q95_hex,
                "nrmse_denominator": denominator,
                "nrmse_denominator_float_hex": denominator_hex,
                "r2_boundary_equal_0_9925": False,
                "nrmse_boundary_equal_0_025": False,
                "strict_stable_margin_pass": False,
                "endpoint_pooled_verdict": verdict,
            }
            stability_records.extend(
                empty_stability_records(
                    diagnostic,
                    diagnostic_kind="ENDPOINT_VETO",
                    denominator=pooled.denominator,
                    reason=reason,
                )
            )
            distribution_records.extend(
                empty_distribution_records(diagnostic, reason=reason)
            )
            endpoint_results.append(
                {
                    "diagnostic": diagnostic,
                    "verdict": verdict,
                    "source_component_estimator_binding": binding,
                    "pooled_stable_margin_pass": None,
                    "cell_stability_pass": None,
                    "distribution_stability_pass": None,
                    "boundary_equality_flags": [],
                    "clear_pass": False,
                    "ambiguity_reason": verdict if source_tie else None,
                }
            )
            pooled_records.append(pooled_record)
            continue

        prediction = site_prediction[diagnostic]
        assert prediction is not None
        metric = regression_metrics(actual, prediction, pooled.denominator)
        metric.require_valid()
        assert metric.r2 is not None and metric.nrmse is not None
        pooled_scope = f"ENDPOINT/{diagnostic}/POOLED"
        endpoint_witnesses = [
            make_threshold_witness(
                cutoff_name="INDEPENDENT_R2_STABLE_MARGIN_0.9925",
                scope=pooled_scope,
                diagnostic=diagnostic,
                value=metric.r2,
                cutoff=INDEPENDENT_MARGIN_R2_CUTOFF,
            ),
            make_threshold_witness(
                cutoff_name="INDEPENDENT_NRMSE_STABLE_MARGIN_0.025",
                scope=pooled_scope,
                diagnostic=diagnostic,
                value=metric.nrmse,
                cutoff=INDEPENDENT_MARGIN_NRMSE_CUTOFF,
            ),
        ]
        stability = evaluate_diagnostic_stability(
            site,
            actual,
            prediction,
            diagnostic=diagnostic,
            diagnostic_kind="ENDPOINT_VETO",
            pooled_denominator=pooled.denominator,
        )
        distribution = evaluate_diagnostic_distribution(
            site, actual, diagnostic=diagnostic
        )
        endpoint_witnesses.extend(stability.witnesses)
        endpoint_witnesses.extend(distribution.witnesses)
        endpoint_witnesses.sort(key=_threshold_sort_key)
        boundary_flags = [
            cutoff
            for cutoff in CUTOFF_NAMES
            if any(
                item["cutoff_name"] == cutoff and item["boundary_equal"]
                for item in endpoint_witnesses
            )
        ]
        pooled_boundary = any(
            item["boundary_equal"] for item in endpoint_witnesses[:2]
        )
        cell_boundary = any(
            item["boundary_equal"] for item in stability.witnesses
        )
        distribution_boundary = any(
            item["boundary_equal"] for item in distribution.witnesses
        )
        pooled_pass = bool(
            np.float64(metric.r2) < np.float64(INDEPENDENT_MARGIN_R2_CUTOFF)
            and np.float64(metric.nrmse)
            > np.float64(INDEPENDENT_MARGIN_NRMSE_CUTOFF)
        )
        if boundary_flags:
            verdict = "AMBIGUOUS_THRESHOLD_EQUALITY"
        elif pooled_pass and stability.gate.all_pass and distribution.passes:
            verdict = "PASS_ENDPOINT_NONREDUNDANT_MARGIN"
        else:
            verdict = "CLEAR_ENDPOINT_VETO_FAILURE"
        sse, sse_hex = _value_and_hex(metric.sse)
        rmse, rmse_hex = _value_and_hex(metric.rmse)
        r2, r2_hex = _value_and_hex(metric.r2)
        nrmse, nrmse_hex = _value_and_hex(metric.nrmse)
        q05, q05_hex = persist_float(pooled.q05)
        q95, q95_hex = persist_float(pooled.q95)
        denominator, denominator_hex = persist_float(pooled.denominator)
        pooled_record = {
            "diagnostic": diagnostic,
            "rows": metric.rows,
            "valid": metric.valid,
            "invalid_reason": metric.invalid_reason,
            "source_component_estimator_binding": binding,
            "sse": sse,
            "sse_float_hex": sse_hex,
            "rmse": rmse,
            "rmse_float_hex": rmse_hex,
            "r2": r2,
            "r2_float_hex": r2_hex,
            "nrmse": nrmse,
            "nrmse_float_hex": nrmse_hex,
            "actual_q05": q05,
            "actual_q05_float_hex": q05_hex,
            "actual_q95": q95,
            "actual_q95_float_hex": q95_hex,
            "nrmse_denominator": denominator,
            "nrmse_denominator_float_hex": denominator_hex,
            "r2_boundary_equal_0_9925": _equal(
                metric.r2, INDEPENDENT_MARGIN_R2_CUTOFF
            ),
            "nrmse_boundary_equal_0_025": _equal(
                metric.nrmse, INDEPENDENT_MARGIN_NRMSE_CUTOFF
            ),
            "strict_stable_margin_pass": pooled_pass and not pooled_boundary,
            "endpoint_pooled_verdict": verdict,
        }
        endpoint_results.append(
            {
                "diagnostic": diagnostic,
                "verdict": verdict,
                "source_component_estimator_binding": binding,
                "pooled_stable_margin_pass": (
                    None if pooled_boundary else pooled_pass
                ),
                "cell_stability_pass": (
                    None if cell_boundary else stability.gate.all_pass
                ),
                "distribution_stability_pass": (
                    None if distribution_boundary else distribution.passes
                ),
                "boundary_equality_flags": boundary_flags,
                "clear_pass": verdict == "PASS_ENDPOINT_NONREDUNDANT_MARGIN",
                "ambiguity_reason": (
                    verdict
                    if verdict == "AMBIGUOUS_THRESHOLD_EQUALITY"
                    else None
                ),
            }
        )
        pooled_records.append(pooled_record)
        stability_records.extend(stability.records)
        distribution_records.extend(distribution.records)
        witnesses.extend(endpoint_witnesses)
    if (
        len(endpoint_results) != 3
        or len(pooled_records) != 3
        or len(stability_records) != 87
        or len(distribution_records) != 156
    ):
        raise AssertionError("endpoint evidence record count drift")
    if any(tuple(record) != ENDPOINT_POOLED_KEYS for record in pooled_records):
        raise AssertionError("endpoint pooled record schema drift")
    return DerivedEvaluation(
        site_actual={key: value for key, value in site_actual.items() if value is not None},
        site_prediction=site_prediction,
        group_metadata=group_metadata,
        group_actual={key: value for key, value in group_actual.items() if value is not None},
        group_prediction=group_prediction,
        endpoint_results=tuple(endpoint_results),
        endpoint_pooled_records=tuple(pooled_records),
        stability_records=tuple(stability_records),
        distribution_records=tuple(distribution_records),
        threshold_witnesses=tuple(witnesses),
    )


def build_oof_arrow_table(
    prepared: PreparedExecution,
    raw_evaluations: Mapping[str, ResponseEvaluation],
) -> tuple[Any, Any]:
    import pandas as pd
    import pyarrow as pa

    np = _np()
    site = prepared.joined.site
    valid_time = site["valid_time_utc"].to_numpy(dtype=np.int64, copy=True)
    operating_day = pd.to_datetime(
        site["target_operating_day_kst__external"], errors="raise"
    ).dt.date.to_numpy()
    forecast_hour = site["forecast_hour"].to_numpy(dtype=np.int16, copy=True)
    site_id = site["site_id"].to_numpy(dtype=np.int16, copy=True)
    group = site["group__external"].astype(str).to_numpy()
    capacity = site["capacity_mw__external"].to_numpy(
        dtype=np.float64, copy=True
    )
    recomputed_fold_ordinals = assign_fold_ordinals(
        site["target_operating_day_kst__external"]
    )
    if not bool(
        np.array_equal(
            recomputed_fold_ordinals, prepared.joined.fold_ordinals
        )
    ):
        raise IntegrityError("OOF fold differs from target operating day")
    fold_names = np.asarray(
        [FOLDS[int(value) - 1].name for value in recomputed_fold_ordinals],
        dtype=object,
    )
    columns: dict[str, list[Any]] = {name: [] for name in OOF_COLUMNS}
    estimators = ("RIDGE_PIPELINE", "EXTRA_TREES")
    expected_selected_rows = 0
    for response_ordinal, response in enumerate(RESPONSES, start=1):
        evaluation = raw_evaluations[response]
        if evaluation.independent_estimator is not None:
            if evaluation.independent_estimator not in estimators:
                raise IntegrityError("OOF independent estimator differs")
            expected_selected_rows += EXPECTED_SITE_ROWS
        for estimator_ordinal, estimator_name in enumerate(estimators, start=1):
            prediction = (
                evaluation.ridge_oof
                if estimator_name == "RIDGE_PIPELINE"
                else evaluation.extra_trees_oof
            )
            skipped = prediction is None
            reason = evaluation.skip_reason if skipped else None
            if skipped and reason is None:
                raise IntegrityError("skipped OOF lacks exact reason")
            columns["planned_oof_slot_id"].extend(
                planned_oof_slot_id(
                    response_ordinal, estimator_ordinal, row_ordinal
                )
                for row_ordinal in range(1, EXPECTED_SITE_ROWS + 1)
            )
            columns["response"].extend([response] * EXPECTED_SITE_ROWS)
            columns["estimator"].extend(
                [estimator_name] * EXPECTED_SITE_ROWS
            )
            columns["fold"].extend(fold_names.tolist())
            columns["valid_time_utc"].extend(valid_time.tolist())
            columns["target_operating_day_kst"].extend(operating_day.tolist())
            columns["forecast_hour"].extend(forecast_hour.tolist())
            columns["site_id"].extend(site_id.tolist())
            columns["group"].extend(group.tolist())
            columns["capacity_mw"].extend(capacity.tolist())
            columns["actual"].extend(evaluation.actual.tolist())
            if skipped:
                columns["oof_prediction"].extend([None] * EXPECTED_SITE_ROWS)
                columns["residual"].extend([None] * EXPECTED_SITE_ROWS)
            else:
                predicted = _float64_vector(prediction, "OOF output prediction")
                columns["oof_prediction"].extend(predicted.tolist())
                columns["residual"].extend(
                    np.ascontiguousarray(
                        evaluation.actual - predicted, dtype=np.float64
                    ).tolist()
                )
            columns["slot_status"].extend(
                [UNIT_SKIPPED if skipped else UNIT_COMPLETED]
                * EXPECTED_SITE_ROWS
            )
            columns["invalid_reason"].extend(
                [reason] * EXPECTED_SITE_ROWS
            )
            columns["independent_selected"].extend(
                [
                    estimator_name == evaluation.independent_estimator
                    and not skipped
                ]
                * EXPECTED_SITE_ROWS
            )
    if any(len(values) != EXPECTED_OOF_ROWS for values in columns.values()):
        raise AssertionError("OOF column row count drift")
    if sum(bool(value) for value in columns["independent_selected"]) != (
        expected_selected_rows
    ):
        raise IntegrityError("OOF independent-selected row count differs")
    schema = pa.schema(
        [
            pa.field("planned_oof_slot_id", pa.string(), nullable=False),
            pa.field("response", pa.string(), nullable=False),
            pa.field("estimator", pa.string(), nullable=False),
            pa.field("fold", pa.string(), nullable=False),
            pa.field(
                "valid_time_utc",
                pa.timestamp("ns", tz="UTC"),
                nullable=False,
            ),
            pa.field("target_operating_day_kst", pa.date32(), nullable=False),
            pa.field("forecast_hour", pa.int16(), nullable=False),
            pa.field("site_id", pa.int16(), nullable=False),
            pa.field("group", pa.string(), nullable=False),
            pa.field("capacity_mw", pa.float64(), nullable=False),
            pa.field("actual", pa.float64(), nullable=True),
            pa.field("oof_prediction", pa.float64(), nullable=True),
            pa.field("residual", pa.float64(), nullable=True),
            pa.field("slot_status", pa.string(), nullable=False),
            pa.field("invalid_reason", pa.string(), nullable=True),
            pa.field("independent_selected", pa.bool_(), nullable=False),
        ]
    )
    arrays = [
        pa.array(columns[field.name], type=field.type)
        for field in schema
    ]
    return schema, arrays


def build_derived_arrow_table(
    prepared: PreparedExecution,
    raw_evaluations: Mapping[str, ResponseEvaluation],
    derived: DerivedEvaluation,
) -> tuple[Any, Any]:
    import pandas as pd
    import pyarrow as pa

    np = _np()
    site = prepared.joined.site
    site_valid = site["valid_time_utc"].to_numpy(dtype=np.int64, copy=True)
    site_days = pd.to_datetime(
        site["target_operating_day_kst__external"], errors="raise"
    ).dt.date.to_numpy()
    site_fh = site["forecast_hour"].to_numpy(dtype=np.int16, copy=True)
    site_ids = site["site_id"].to_numpy(dtype=np.int16, copy=True)
    site_groups = site["group__external"].astype(str).to_numpy()
    site_capacity = site["capacity_mw__external"].to_numpy(
        dtype=np.float64, copy=True
    )
    group_meta = derived.group_metadata
    valid_to_day = (
        pd.DataFrame(
            {
                "valid_time_utc": site["valid_time_utc"],
                "day": site["target_operating_day_kst__external"],
                "forecast_hour": site["forecast_hour"],
            }
        )
        .drop_duplicates("valid_time_utc")
        .set_index("valid_time_utc")
    )
    group_valid = group_meta["valid_time_utc"].to_numpy(
        dtype=np.int64, copy=True
    )
    group_days = pd.to_datetime(
        group_meta["valid_time_utc"].map(valid_to_day["day"]), errors="raise"
    ).dt.date.to_numpy()
    group_fh = group_meta["valid_time_utc"].map(
        valid_to_day["forecast_hour"]
    ).to_numpy(dtype=np.int16)
    group_groups = group_meta["group"].astype(str).to_numpy()
    group_capacity = group_meta["capacity_mw"].to_numpy(
        dtype=np.float64, copy=True
    )
    columns: dict[str, list[Any]] = {name: [] for name in DERIVED_COLUMNS}
    for diagnostic_ordinal, diagnostic in enumerate(
        DERIVED_DIAGNOSTICS, start=1
    ):
        dependencies = _derived_dependencies(diagnostic)
        binding = {
            response: raw_evaluations[response].independent_estimator
            for response in dependencies
        }
        binding_json = strict_json_dumps(binding, pretty=False).decode("ascii")
        for (
            scope,
            rows,
            valid_times,
            days,
            forecast_hours,
            site_values,
            groups,
            capacities,
            actual,
            prediction,
        ) in (
            (
                "SITE",
                EXPECTED_SITE_ROWS,
                site_valid,
                site_days,
                site_fh,
                site_ids,
                site_groups,
                site_capacity,
                derived.site_actual[diagnostic],
                derived.site_prediction[diagnostic],
            ),
            (
                "GROUP",
                EXPECTED_GROUP_ROWS,
                group_valid,
                group_days,
                group_fh,
                None,
                group_groups,
                group_capacity,
                derived.group_actual[diagnostic],
                derived.group_prediction[diagnostic],
            ),
        ):
            columns["planned_derived_slot_id"].extend(
                planned_derived_slot_id(
                    diagnostic_ordinal, scope, row_ordinal
                )
                for row_ordinal in range(1, rows + 1)
            )
            columns["diagnostic"].extend([diagnostic] * rows)
            columns["formula_id"].extend([diagnostic] * rows)
            columns["scope_type"].extend([scope] * rows)
            columns["valid_time_utc"].extend(valid_times.tolist())
            columns["target_operating_day_kst"].extend(days.tolist())
            columns["forecast_hour"].extend(forecast_hours.tolist())
            columns["site_id"].extend(
                site_values.tolist() if site_values is not None else [None] * rows
            )
            columns["group"].extend(groups.tolist())
            columns["capacity_mw"].extend(capacities.tolist())
            actual_values = _float64_vector(actual, "derived output actual")
            columns["actual"].extend(actual_values.tolist())
            if prediction is None:
                columns["oof_prediction"].extend([None] * rows)
                columns["residual"].extend([None] * rows)
                columns["valid"].extend([False] * rows)
                columns["invalid_reason"].extend(
                    ["STRUCTURALLY_NOT_APPLICABLE"] * rows
                )
            else:
                predicted = _float64_vector(
                    prediction, "derived output prediction"
                )
                columns["oof_prediction"].extend(predicted.tolist())
                columns["residual"].extend(
                    np.ascontiguousarray(
                        actual_values - predicted, dtype=np.float64
                    ).tolist()
                )
                columns["valid"].extend([True] * rows)
                columns["invalid_reason"].extend([None] * rows)
            columns["endpoint_veto_required"].extend(
                [diagnostic in ENDPOINT_VETO_DIAGNOSTICS] * rows
            )
            columns["source_component_estimator_binding_json"].extend(
                [binding_json] * rows
            )
    if any(len(values) != EXPECTED_DERIVED_ROWS for values in columns.values()):
        raise AssertionError("derived output column row count drift")
    schema = pa.schema(
        [
            pa.field("planned_derived_slot_id", pa.string(), nullable=False),
            pa.field("diagnostic", pa.string(), nullable=False),
            pa.field("formula_id", pa.string(), nullable=False),
            pa.field("scope_type", pa.string(), nullable=False),
            pa.field(
                "valid_time_utc",
                pa.timestamp("ns", tz="UTC"),
                nullable=False,
            ),
            pa.field("target_operating_day_kst", pa.date32(), nullable=False),
            pa.field("forecast_hour", pa.int16(), nullable=False),
            pa.field("site_id", pa.int16(), nullable=True),
            pa.field("group", pa.string(), nullable=False),
            pa.field("capacity_mw", pa.float64(), nullable=False),
            pa.field("actual", pa.float64(), nullable=True),
            pa.field("oof_prediction", pa.float64(), nullable=True),
            pa.field("residual", pa.float64(), nullable=True),
            pa.field("valid", pa.bool_(), nullable=False),
            pa.field("invalid_reason", pa.string(), nullable=True),
            pa.field("endpoint_veto_required", pa.bool_(), nullable=False),
            pa.field(
                "source_component_estimator_binding_json",
                pa.string(),
                nullable=False,
            ),
        ]
    )
    arrays = [
        pa.array(columns[field.name], type=field.type)
        for field in schema
    ]
    return schema, arrays


def _validate_joined_metadata_and_schedule(site: Any, group: Any) -> None:
    """Freeze all site/group metadata and D-2 schedule relations pre-fit."""

    import pandas as pd

    np = _np()
    expected_groups = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
    expected_group_site_counts = {
        "kpx_group_1": 6,
        "kpx_group_2": 6,
        "kpx_group_3": 5,
    }
    if (
        int(site["valid_time_utc"].nunique()) != EXPECTED_TIMESTAMPS
        or int(group["valid_time_utc"].nunique()) != EXPECTED_TIMESTAMPS
        or not bool(
            (site.groupby("valid_time_utc", sort=False).size() == 17).all()
        )
        or not bool(
            (group.groupby("valid_time_utc", sort=False).size() == 3).all()
        )
        or set(site["site_id"].unique()) != set(range(1, 18))
        or set(site["group__external"].unique()) != set(expected_groups)
        or set(group["group"].unique()) != set(expected_groups)
    ):
        raise IntegrityError("site/group timestamp or member cardinality differs")
    site_membership = site.loc[
        :, ["site_id", "group__external"]
    ].drop_duplicates()
    if (
        len(site_membership) != 17
        or bool(site_membership["site_id"].duplicated().any())
        or site_membership.groupby("group__external").size().to_dict()
        != expected_group_site_counts
    ):
        raise IntegrityError("site-to-group membership differs")
    for column in (
        "target_operating_day_kst",
        "site_count",
        "capacity_mw",
    ):
        left = group[f"{column}__external"].to_numpy()
        right = group[f"{column}__predictor"].to_numpy()
        if not bool(np.array_equal(left, right)):
            raise IntegrityError(f"group metadata parity differs: {column}")
    expected_site_count_vector = group["group"].map(
        expected_group_site_counts
    ).to_numpy(dtype=np.int64)
    if not bool(
        np.array_equal(
            group["site_count__external"].to_numpy(dtype=np.int64),
            expected_site_count_vector,
        )
    ):
        raise IntegrityError("group site_count differs from frozen membership")

    def schedule(frame: Any, *, label: str) -> None:
        target_days = pd.to_datetime(
            frame["target_operating_day_kst__external"], errors="raise"
        )
        forecast_local = pd.to_datetime(
            frame["forecast_kst_dtm"], errors="raise"
        )
        available_local = pd.to_datetime(
            frame["data_available_kst_dtm"], errors="raise"
        )
        for values, name in (
            (target_days, "target operating day"),
            (forecast_local, "forecast KST"),
            (available_local, "data-available KST"),
        ):
            if bool(values.isna().any()) or values.dt.tz is not None:
                raise IntegrityError(f"{label} {name} timezone/null differs")
        if not bool((target_days == target_days.dt.normalize()).all()):
            raise IntegrityError(f"{label} target operating day is not midnight")
        expected_days = pd.DatetimeIndex(
            [
                pd.Timestamp(year=year, month=month, day=day)
                for year in (2022, 2023)
                for month in range(1, 13)
                for day in (5, 20)
            ]
        )
        observed_days = pd.DatetimeIndex(target_days.unique()).sort_values()
        if not observed_days.equals(expected_days):
            raise IntegrityError(f"{label} exact 48-day sample differs")
        hour_offset = (
            forecast_local - (target_days + pd.Timedelta(hours=1))
        ) / pd.Timedelta(hours=1)
        offsets = hour_offset.to_numpy(dtype=np.float64)
        forecast_hours = frame["forecast_hour"].to_numpy(dtype=np.int64)
        if (
            not bool(np.isfinite(offsets).all())
            or not bool(np.equal(offsets, np.floor(offsets)).all())
            or bool(np.any(offsets < 0.0))
            or bool(np.any(offsets > 23.0))
            or not bool(
                np.array_equal(
                    forecast_hours,
                    offsets.astype(np.int64, copy=False) + 28,
                )
            )
        ):
            raise IntegrityError(f"{label} forecast-hour f028..f051 differs")
        expected_available = (
            target_days - pd.Timedelta(days=1) + pd.Timedelta(hours=13)
        )
        if not bool((available_local == expected_available).all()):
            raise IntegrityError(f"{label} D-1 13KST availability differs")
        run_utc = pd.to_datetime(frame["run_init_utc"], utc=True, errors="raise")
        expected_run_utc = (
            target_days - pd.Timedelta(days=2) + pd.Timedelta(hours=12)
        ).dt.tz_localize("UTC")
        valid_ns = frame["valid_time_utc"].to_numpy(dtype=np.int64)
        if not bool(
            np.array_equal(
                normalize_utc_int64_ns(run_utc),
                normalize_utc_int64_ns(expected_run_utc),
            )
        ):
            raise IntegrityError(f"{label} D-2 12Z run differs")
        forecast_utc = forecast_local.dt.tz_localize(
            "Asia/Seoul", ambiguous="raise", nonexistent="raise"
        ).dt.tz_convert("UTC")
        if not bool(
            np.array_equal(normalize_utc_int64_ns(forecast_utc), valid_ns)
        ):
            raise IntegrityError(f"{label} forecast KST/valid UTC differs")
        expected_valid_ns = normalize_utc_int64_ns(run_utc) + (
            forecast_hours.astype(np.int64, copy=False)
            * np.int64(3_600_000_000_000)
        )
        if not bool(np.array_equal(expected_valid_ns, valid_ns)):
            raise IntegrityError(f"{label} run plus forecast-hour differs")

    schedule(site, label="site")
    schedule(group, label="group")


def load_and_join_bound_inputs(
    artifact_root: Path,
    bound_paths: Mapping[str, Path],
    *,
    authorization: PredataAuthorization,
) -> JoinedInputs:
    """Schema-check, read, exact-key join, and parity-check the four matrices."""

    np = _np()
    authorization.assert_arrow_handoff()
    root = artifact_root.resolve(strict=True)
    provenance_alias = root / (
        "modeling/target_free_multiseason_v5/PROVENANCE_LEDGER.parquet"
    )
    _validate_provenance_alias_schema(authorization, provenance_alias)

    external_site_types = (
        "large_string",
        "large_string",
        "large_string",
        "int64",
        "int64",
        "large_string",
        "float64",
        "float64",
        "float64",
        *("float64" for _ in RESPONSES),
    )
    external_group_types = (
        "large_string",
        "large_string",
        "large_string",
        "int64",
        "large_string",
        "int64",
        "float64",
        *("float64" for _ in RESPONSES),
    )
    predictor_site_types = (
        "large_string",
        "large_string",
        "large_string",
        "large_string",
        "int16",
        "large_string",
        "float64",
        "float64",
        "float64",
        *("float64" for _ in PREDICTORS),
    )
    predictor_group_types = (
        "large_string",
        "large_string",
        "large_string",
        "large_string",
        "large_string",
        "int16",
        "float64",
        *("float64" for _ in PREDICTORS),
    )
    layouts = (
        (
            bound_paths["decoded_site_matrix"],
            EXTERNAL_SITE_COLUMNS,
            external_site_types,
            EXPECTED_SITE_ROWS,
        ),
        (
            bound_paths["decoded_group_matrix"],
            EXTERNAL_GROUP_COLUMNS,
            external_group_types,
            EXPECTED_GROUP_ROWS,
        ),
        (
            bound_paths["bounded_site_predictors"],
            PREDICTOR_SITE_COLUMNS,
            predictor_site_types,
            EXPECTED_SITE_ROWS,
        ),
        (
            bound_paths["bounded_group_predictors"],
            PREDICTOR_GROUP_COLUMNS,
            predictor_group_types,
            EXPECTED_GROUP_ROWS,
        ),
    )
    for path, columns, types, rows in layouts:
        _validate_parquet_names_types_rows(
            authorization,
            path,
            expected_columns=columns,
            expected_type_names=types,
            expected_rows=rows,
        )

    def read_frame(path: Path, columns: Sequence[str]) -> Any:
        table = guarded_read_table(
            authorization, path, columns=list(columns)
        )
        try:
            return table.to_pandas()
        finally:
            del table

    external_site = read_frame(
        bound_paths["decoded_site_matrix"], EXTERNAL_SITE_COLUMNS
    )
    external_group = read_frame(
        bound_paths["decoded_group_matrix"], EXTERNAL_GROUP_COLUMNS
    )
    predictor_site = read_frame(
        bound_paths["bounded_site_predictors"], PREDICTOR_SITE_COLUMNS
    )
    predictor_group = read_frame(
        bound_paths["bounded_group_predictors"], PREDICTOR_GROUP_COLUMNS
    )
    require_exact_columns(external_site, EXTERNAL_SITE_COLUMNS, "external site")
    require_exact_columns(external_group, EXTERNAL_GROUP_COLUMNS, "external group")
    require_exact_columns(predictor_site, PREDICTOR_SITE_COLUMNS, "predictor site")
    require_exact_columns(
        predictor_group, PREDICTOR_GROUP_COLUMNS, "predictor group"
    )
    site = exact_one_to_one_join(
        external_site,
        predictor_site,
        key=SITE_KEY,
        expected_rows=EXPECTED_SITE_ROWS,
    )
    group = exact_one_to_one_join(
        external_group,
        predictor_group,
        key=GROUP_KEY,
        expected_rows=EXPECTED_GROUP_ROWS,
    )
    del external_site, predictor_site
    site = site.sort_values(
        ["valid_time_utc", "site_id"], kind="mergesort", ignore_index=True
    )
    group = group.sort_values(
        ["valid_time_utc", "group"], kind="mergesort", ignore_index=True
    )
    parity_pairs = (
        ("target_operating_day_kst__external", "target_operating_day_kst__predictor"),
        ("group__external", "group__predictor"),
        ("latitude__external", "latitude__predictor"),
        ("longitude__external", "longitude__predictor"),
        ("capacity_mw__external", "capacity_mw__predictor"),
    )
    for left, right in parity_pairs:
        if left not in site or right not in site:
            raise IntegrityError(f"site parity field missing: {left}/{right}")
        if not bool(np.array_equal(site[left].to_numpy(), site[right].to_numpy())):
            raise IntegrityError(f"site metadata parity differs: {left}/{right}")
    _validate_joined_metadata_and_schedule(site, group)

    site_predictor_view = site.loc[
        :,
        [
            "valid_time_utc",
            "group__external",
            "capacity_mw__external",
            *PREDICTORS,
        ],
    ].rename(
        columns={
            "group__external": "group",
            "capacity_mw__external": "capacity_mw",
        }
    )
    aggregated_predictors = capacity_aggregate_raw_uv(
        site_predictor_view, value_columns=PREDICTORS
    )
    group_predictor_sorted = predictor_group.copy()
    del predictor_group
    group_predictor_sorted["valid_time_utc"] = normalize_utc_int64_ns(
        group_predictor_sorted["valid_time_utc"]
    )
    group_predictor_sorted = group_predictor_sorted.sort_values(
        ["valid_time_utc", "group"], kind="mergesort", ignore_index=True
    )
    if not np.array_equal(
        aggregated_predictors[["valid_time_utc", "group"]].to_numpy(),
        group_predictor_sorted[["valid_time_utc", "group"]].to_numpy(),
    ):
        raise IntegrityError("group predictor parity keys differ")
    group_predictor_values = group_predictor_sorted.loc[
        :, PREDICTORS
    ].to_numpy(dtype=np.float64)
    group_predictor_capacity = group_predictor_sorted[
        "capacity_mw"
    ].to_numpy(dtype=np.float64)
    if (
        not bool(np.isfinite(group_predictor_values).all())
        or not bool(np.isfinite(group_predictor_capacity).all())
        or bool(np.any(group_predictor_capacity <= np.float64(0.0)))
    ):
        raise IntegrityError("group predictor values or capacity are non-finite")
    predictor_diff = np.abs(
        aggregated_predictors.loc[:, PREDICTORS].to_numpy(dtype=np.float64)
        - group_predictor_values
    )
    predictor_capacity_diff = np.abs(
        aggregated_predictors["capacity_mw"].to_numpy(dtype=np.float64)
        - group_predictor_capacity
    )
    if bool(np.any(predictor_diff > np.float64(1e-9))) or bool(
        np.any(predictor_capacity_diff > np.float64(1e-9))
    ):
        raise IntegrityError("group predictor aggregation parity exceeds 1e-9")

    site_external_view = site.loc[
        :,
        [
            "valid_time_utc",
            "group__external",
            "capacity_mw__external",
            *RESPONSES,
        ],
    ].rename(
        columns={
            "group__external": "group",
            "capacity_mw__external": "capacity_mw",
        }
    )
    aggregated_external = capacity_aggregate_raw_uv(
        site_external_view, value_columns=RESPONSES
    )
    group_external_sorted = external_group.copy()
    del external_group
    group_external_sorted["valid_time_utc"] = normalize_utc_int64_ns(
        group_external_sorted["valid_time_utc"]
    )
    group_external_sorted = group_external_sorted.sort_values(
        ["valid_time_utc", "group"], kind="mergesort", ignore_index=True
    )
    group_external_values = group_external_sorted.loc[
        :, RESPONSES
    ].to_numpy(dtype=np.float64)
    group_external_capacity = group_external_sorted[
        "capacity_mw"
    ].to_numpy(dtype=np.float64)
    if (
        not bool(np.isfinite(group_external_values).all())
        or not bool(np.isfinite(group_external_capacity).all())
        or bool(np.any(group_external_capacity <= np.float64(0.0)))
    ):
        raise IntegrityError("decoded group values or capacity are non-finite")
    external_diff = np.abs(
        aggregated_external.loc[:, RESPONSES].to_numpy(dtype=np.float64)
        - group_external_values
    )
    external_capacity_diff = np.abs(
        aggregated_external["capacity_mw"].to_numpy(dtype=np.float64)
        - group_external_capacity
    )
    if bool(np.any(external_diff > np.float64(1e-9))) or bool(
        np.any(external_capacity_diff > np.float64(1e-9))
    ):
        raise IntegrityError("decoded group aggregation parity exceeds 1e-9")

    matrix = np.ascontiguousarray(
        site.loc[:, PREDICTORS].to_numpy(dtype=np.float64), dtype=np.float64
    )
    if matrix.shape != (EXPECTED_SITE_ROWS, len(PREDICTORS)):
        raise IntegrityError("site predictor matrix shape differs")
    if not bool(np.isfinite(matrix).all()):
        raise IntegrityError("site predictor matrix is non-finite")
    fold_ordinals = assign_fold_ordinals(
        site["target_operating_day_kst__external"]
    )
    return JoinedInputs(site, group, matrix, fold_ordinals)


@dataclasses.dataclass(slots=True)
class PredataAuthorization:
    """State machine guarding all logical decoded and predictor value access."""

    validation: ControlValidation
    artifact_root: Path | None = None
    threadpool_recorder: ThreadpoolCaptureRecorder | None = None
    alias_files_published: bool = False
    alias_metadata_validated: bool = False
    bound_input_identities_rehashed: bool = False
    source_metadata_scope_open: bool = False
    source_footer_metadata_validated: bool = False
    arrow_handoff_started: bool = False

    def assert_source_metadata_ready(self) -> None:
        """Authorize only post-GO source footer/census metadata before root."""

        required_true = (
            "amendment_pass",
            "code_seal_pass",
            "execution_authorization_pass",
            "independent_review_pass",
            "independent_go_pass",
            "runtime_identity_pass",
            "output_namespace_absent_pass",
        )
        failed = [
            name for name in required_true if not bool(getattr(self.validation, name))
        ]
        if failed:
            raise PredataAuthorizationError(
                "source metadata authorization failed: " + ",".join(failed)
            )
        if (
            self.artifact_root is None
            or not self.bound_input_identities_rehashed
            or not self.source_metadata_scope_open
            or self.source_footer_metadata_validated
            or self.validation.bound_input_identity_and_schema_pass
            or self.validation.aliases_pass
            or self.alias_files_published
            or self.arrow_handoff_started
        ):
            raise PredataAuthorizationError(
                "source metadata scope state differs"
            )

    def assert_alias_publication_ready(self) -> None:
        failed = [
            field.name
            for field in dataclasses.fields(self.validation)
            if field.name != "aliases_pass"
            and not bool(getattr(self.validation, field.name))
        ]
        if failed:
            raise PredataAuthorizationError(
                "alias publication authorization failed: " + ",".join(failed)
            )

    def assert_ready(self) -> None:
        failed = [
            field.name
            for field in dataclasses.fields(self.validation)
            if not bool(getattr(self.validation, field.name))
        ]
        if failed:
            raise PredataAuthorizationError(
                "predata authorization failed: " + ",".join(failed)
            )

    def assert_alias_metadata_ready(self) -> None:
        self.assert_alias_publication_ready()
        if not self.alias_files_published or self.artifact_root is None:
            raise PredataAuthorizationError(
                "final alias files or exact artifact root are unavailable"
            )

    def authorize_arrow_handoff(self) -> None:
        self.assert_ready()
        if not self.alias_metadata_validated:
            raise PredataAuthorizationError(
                "final alias metadata closure is not validated"
            )
        self.arrow_handoff_started = True

    def assert_arrow_handoff(self) -> None:
        if not self.arrow_handoff_started:
            raise PredataAuthorizationError(
                "logical value API called before authorized Arrow handoff"
            )
        self.assert_ready()

    def capture_threadpool(self, *, phase: str, label: str) -> None:
        if self.threadpool_recorder is None:
            raise PredataAuthorizationError(
                "threadpool recorder is unavailable"
            )
        self.threadpool_recorder.capture(phase=phase, label=label)


def guarded_parquet_file(
    authorization: PredataAuthorization, path: Path, *args: Any, **kwargs: Any
) -> Any:
    authorization.assert_arrow_handoff()
    import pyarrow.parquet as pq

    return pq.ParquetFile(path, *args, **kwargs)


def guarded_alias_parquet_file(
    authorization: PredataAuthorization,
    path: Path,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Permit only final provenance footer metadata before value handoff."""

    authorization.assert_alias_metadata_ready()
    assert authorization.artifact_root is not None
    _assert_no_linklike_absolute(path, "final provenance alias")
    expected = (
        authorization.artifact_root.resolve(strict=True)
        / OUTPUT_ROOT_RELATIVE
        / "PROVENANCE_LEDGER.parquet"
    ).resolve(strict=True)
    observed = path.resolve(strict=True)
    if observed != expected:
        raise PredataAuthorizationError(
            "pre-handoff Parquet metadata path is not final provenance alias"
        )
    import pyarrow.parquet as pq

    return pq.ParquetFile(observed, *args, **kwargs)


def guarded_source_parquet_file(
    authorization: PredataAuthorization,
    path: Path,
    *args: Any,
    allowed_paths: Sequence[Path],
    **kwargs: Any,
) -> Any:
    """Permit only whitelisted source Parquet footers before V5 root creation."""

    authorization.assert_source_metadata_ready()
    assert authorization.artifact_root is not None
    _assert_no_linklike_absolute(path, "pre-root source Parquet")
    observed = path.resolve(strict=True)
    allowed = tuple(item.resolve(strict=True) for item in allowed_paths)
    if observed not in allowed:
        raise PredataAuthorizationError(
            "pre-root source Parquet path is outside the exact metadata scope"
        )
    import pyarrow.parquet as pq

    return pq.ParquetFile(observed, *args, **kwargs)


def guarded_read_table(
    authorization: PredataAuthorization, path: Path, *args: Any, **kwargs: Any
) -> Any:
    authorization.assert_arrow_handoff()
    import pyarrow.parquet as pq

    return pq.read_table(path, *args, **kwargs)


def guarded_pandas_read_parquet(
    authorization: PredataAuthorization, path: Path, *args: Any, **kwargs: Any
) -> Any:
    authorization.assert_arrow_handoff()
    import pandas as pd

    return pd.read_parquet(path, *args, **kwargs)


def guarded_bound_value_open(
    authorization: PredataAuthorization,
    path: Path,
    mode: str = "rb",
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Guard direct Path.open calls for bound decoded or predictor values."""

    authorization.assert_arrow_handoff()
    if mode != "rb":
        raise PredataAuthorizationError(
            "bound value Path.open is read-only and binary-only"
        )
    return path.open(mode, *args, **kwargs)


def guarded_parquet_schema(
    authorization: PredataAuthorization,
    path: Path,
    *,
    expected_arrow_schema: Any,
    expected_rows: int,
) -> Any:
    authorization.assert_arrow_handoff()
    parquet_file = guarded_parquet_file(authorization, path)
    try:
        observed_schema = parquet_file.schema_arrow
        if not observed_schema.equals(expected_arrow_schema, check_metadata=True):
            raise IntegrityError(f"Parquet Arrow schema differs: {path}")
        if int(parquet_file.metadata.num_rows) != expected_rows:
            raise IntegrityError(f"Parquet row count differs: {path}")
        return observed_schema
    finally:
        parquet_file.close()


def assert_runtime_environment_caps(environment: Mapping[str, str]) -> None:
    for key, expected in ENVIRONMENT_CAPS.items():
        if environment.get(key) != expected:
            raise PredataAuthorizationError(
                f"runtime environment drift: {key}={environment.get(key)!r}"
            )
    if "PYTHONPYCACHEPREFIX" in environment:
        raise PredataAuthorizationError("PYTHONPYCACHEPREFIX must be absent")


AFFINE_CORRELATION_CUTOFF: Final[float] = 0.9999
AFFINE_NRMSE_CUTOFF: Final[float] = 0.01
PARENT_R2_CUTOFF: Final[float] = 0.98
PARENT_NRMSE_CUTOFF: Final[float] = 0.10
INDEPENDENT_DUPLICATE_R2_CUTOFF: Final[float] = 0.995
INDEPENDENT_DUPLICATE_NRMSE_CUTOFF: Final[float] = 0.02
INDEPENDENT_MARGIN_R2_CUTOFF: Final[float] = 0.9925
INDEPENDENT_MARGIN_NRMSE_CUTOFF: Final[float] = 0.025
CELL_R2_CUTOFF: Final[float] = 0.995
CELL_NRMSE_CUTOFF: Final[float] = 0.02
CELL_WEAK_FLOOR_NRMSE_CUTOFF: Final[float] = 0.01
IQR_RATIO_MIN: Final[float] = 0.1
IQR_RATIO_MAX: Final[float] = 10.0
PSI_MAX: Final[float] = 0.5
SST_GUARD: Final[float] = 1e-12
SXX_GUARD: Final[float] = 1e-12
NRMSE_DENOMINATOR_FLOOR: Final[float] = 1e-9
MINIMUM_CELL_ROWS: Final[int] = 48


def _np() -> Any:
    import numpy as np

    return np


def _finite_float64(value: Any, name: str) -> Any:
    np = _np()
    converted = np.float64(value)
    if not bool(np.isfinite(converted)):
        raise IntegrityError(f"{name} must be a finite np.float64")
    return converted


def _float64_vector(
    values: Any, name: str, *, allow_empty: bool = False
) -> Any:
    np = _np()
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise IntegrityError(f"{name} must be one-dimensional")
    if not allow_empty and array.size == 0:
        raise IntegrityError(f"{name} must not be empty")
    if not bool(np.isfinite(array).all()):
        raise IntegrityError(f"{name} contains a non-finite value")
    return array


def canonical_float_hex(value: Any) -> str:
    """Return the exact binary64 witness after persistence-only zero folding."""

    np = _np()
    converted = _finite_float64(value, "float witness")
    if bool(converted == np.float64(0.0)):
        converted = np.float64(0.0)
    return float(converted).hex().lower()


def persist_float(value: Any) -> tuple[float, str]:
    np = _np()
    converted = _finite_float64(value, "persisted float")
    if bool(converted == np.float64(0.0)):
        converted = np.float64(0.0)
    return float(converted), canonical_float_hex(converted)


@dataclasses.dataclass(frozen=True, slots=True)
class MetricResult:
    rows: int
    valid: bool
    invalid_reason: str | None
    unique_count: int
    sst: Any | None
    sse: Any | None
    rmse: Any | None
    r2: Any | None
    nrmse: Any | None
    denominator: Any

    def require_valid(self) -> "MetricResult":
        if not self.valid:
            raise IntegrityError(
                f"metric is invalid: {self.invalid_reason or 'UNKNOWN'}"
            )
        return self


def linear_quantile(values: Any, quantile: float) -> Any:
    np = _np()
    array = _float64_vector(values, "quantile values")
    q = _finite_float64(quantile, "quantile")
    if bool(q < np.float64(0.0)) or bool(q > np.float64(1.0)):
        raise IntegrityError("quantile must be in [0, 1]")
    return np.float64(np.quantile(array, q, method="linear"))


@dataclasses.dataclass(frozen=True, slots=True)
class PooledDenominator:
    q05: Any
    q95: Any
    denominator: Any


def pooled_nrmse_denominator(actual: Any) -> PooledDenominator:
    np = _np()
    array = _float64_vector(actual, "pooled actual")
    q05 = np.float64(np.quantile(array, np.float64(0.05), method="linear"))
    q95 = np.float64(np.quantile(array, np.float64(0.95), method="linear"))
    denominator = np.float64(
        max(np.float64(q95 - q05), np.float64(NRMSE_DENOMINATOR_FLOOR))
    )
    return PooledDenominator(q05=q05, q95=q95, denominator=denominator)


def regression_metrics(
    actual: Any,
    prediction: Any,
    denominator: Any,
    *,
    minimum_rows: int = 1,
    unique_reason: str = "UNUSABLE_CELL_UNIQUE_COUNT_LT_3",
    sst_reason: str = "UNUSABLE_CELL_SST_LTE_1E_MINUS_12",
) -> MetricResult:
    """Compute float64 metrics, using a caller-supplied pooled denominator."""

    np = _np()
    y = _float64_vector(actual, "actual")
    yhat = _float64_vector(prediction, "prediction")
    denom = _finite_float64(denominator, "NRMSE denominator")
    if bool(denom <= np.float64(0.0)):
        raise IntegrityError("NRMSE denominator must be positive")
    if y.shape != yhat.shape:
        raise IntegrityError("actual and prediction shapes differ")
    if y.size < minimum_rows:
        raise IntegrityError(
            f"metric row count {y.size} is below required {minimum_rows}"
        )
    unique_count = int(np.unique(y).size)
    mean_y = np.float64(np.sum(y, dtype=np.float64) / np.float64(y.size))
    centered = np.ascontiguousarray(y - mean_y, dtype=np.float64)
    residual = np.ascontiguousarray(y - yhat, dtype=np.float64)
    sst = np.float64(np.sum(centered * centered, dtype=np.float64))
    sse = np.float64(np.sum(residual * residual, dtype=np.float64))
    if unique_count < 3:
        return MetricResult(
            rows=int(y.size),
            valid=False,
            invalid_reason=unique_reason,
            unique_count=unique_count,
            sst=sst,
            sse=None,
            rmse=None,
            r2=None,
            nrmse=None,
            denominator=denom,
        )
    if bool(sst <= np.float64(SST_GUARD)):
        return MetricResult(
            rows=int(y.size),
            valid=False,
            invalid_reason=sst_reason,
            unique_count=unique_count,
            sst=sst,
            sse=None,
            rmse=None,
            r2=None,
            nrmse=None,
            denominator=denom,
        )
    rmse = np.float64(np.sqrt(np.float64(sse / np.float64(y.size))))
    r2 = np.float64(np.float64(1.0) - np.float64(sse / sst))
    nrmse = np.float64(rmse / denom)
    for name, value in (("sse", sse), ("rmse", rmse), ("r2", r2), ("nrmse", nrmse)):
        _finite_float64(value, name)
    return MetricResult(
        rows=int(y.size),
        valid=True,
        invalid_reason=None,
        unique_count=unique_count,
        sst=sst,
        sse=sse,
        rmse=rmse,
        r2=r2,
        nrmse=nrmse,
        denominator=denom,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class AffineUnitResult:
    intercept: Any
    slope: Any
    sxx: Any
    constant_predictor_branch: bool
    prediction: Any


def analytic_affine_fit_predict(
    train_x: Any, train_y: Any, heldout_x: Any
) -> AffineUnitResult:
    """Fit one closed-form affine unit and predict one held-out fold."""

    np = _np()
    x = _float64_vector(train_x, "affine train predictor")
    y = _float64_vector(train_y, "affine train response")
    held = _float64_vector(heldout_x, "affine heldout predictor")
    if x.shape != y.shape:
        raise IntegrityError("affine train predictor and response shapes differ")
    mean_x = np.float64(np.sum(x, dtype=np.float64) / np.float64(x.size))
    mean_y = np.float64(np.sum(y, dtype=np.float64) / np.float64(y.size))
    dx = np.ascontiguousarray(x - mean_x, dtype=np.float64)
    dy = np.ascontiguousarray(y - mean_y, dtype=np.float64)
    sxx = np.float64(np.sum(dx * dx, dtype=np.float64))
    if bool(sxx <= np.float64(SXX_GUARD)):
        slope = np.float64(0.0)
        intercept = mean_y
        constant = True
    else:
        sxy = np.float64(np.sum(dx * dy, dtype=np.float64))
        slope = np.float64(sxy / sxx)
        intercept = np.float64(mean_y - np.float64(slope * mean_x))
        constant = False
    prediction = np.ascontiguousarray(
        intercept + slope * held, dtype=np.float64
    )
    if not bool(np.isfinite(prediction).all()):
        raise IntegrityError("affine prediction contains a non-finite value")
    return AffineUnitResult(
        intercept=intercept,
        slope=slope,
        sxx=sxx,
        constant_predictor_branch=constant,
        prediction=prediction,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CorrelationResult:
    value: Any | None
    valid: bool
    reason: str | None
    centered_sum_squares_x: Any
    centered_sum_squares_y: Any


def pearson_correlation(x_values: Any, y_values: Any) -> CorrelationResult:
    np = _np()
    x = _float64_vector(x_values, "Pearson x")
    y = _float64_vector(y_values, "Pearson y")
    if x.shape != y.shape:
        raise IntegrityError("Pearson vectors differ in shape")
    mean_x = np.float64(np.sum(x, dtype=np.float64) / np.float64(x.size))
    mean_y = np.float64(np.sum(y, dtype=np.float64) / np.float64(y.size))
    dx = np.ascontiguousarray(x - mean_x, dtype=np.float64)
    dy = np.ascontiguousarray(y - mean_y, dtype=np.float64)
    sxx = np.float64(np.sum(dx * dx, dtype=np.float64))
    syy = np.float64(np.sum(dy * dy, dtype=np.float64))
    if bool(sxx <= np.float64(SXX_GUARD)) or bool(
        syy <= np.float64(SXX_GUARD)
    ):
        return CorrelationResult(
            value=None,
            valid=False,
            reason="DEGENERATE_AFFINE_PREDICTOR_SXX_LTE_1E_MINUS_12",
            centered_sum_squares_x=sxx,
            centered_sum_squares_y=syy,
        )
    numerator = np.float64(np.sum(dx * dy, dtype=np.float64))
    denominator = np.float64(np.sqrt(np.float64(sxx * syy)))
    value = np.float64(numerator / denominator)
    value = np.float64(np.clip(value, np.float64(-1.0), np.float64(1.0)))
    _finite_float64(value, "Pearson correlation")
    return CorrelationResult(value, True, None, sxx, syy)


def spearman_correlation(x_values: Any, y_values: Any) -> CorrelationResult:
    from scipy.stats import rankdata

    np = _np()
    x = _float64_vector(x_values, "Spearman x")
    y = _float64_vector(y_values, "Spearman y")
    if x.shape != y.shape:
        raise IntegrityError("Spearman vectors differ in shape")
    x_rank = np.ascontiguousarray(rankdata(x, method="average"), dtype=np.float64)
    y_rank = np.ascontiguousarray(rankdata(y, method="average"), dtype=np.float64)
    result = pearson_correlation(x_rank, y_rank)
    if not result.valid:
        return CorrelationResult(
            value=None,
            valid=False,
            reason="DEGENERATE_SPEARMAN_RANK_VARIANCE_LTE_1E_MINUS_12",
            centered_sum_squares_x=result.centered_sum_squares_x,
            centered_sum_squares_y=result.centered_sum_squares_y,
        )
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class EstimatorMetric:
    estimator: str
    rmse: Any
    r2: Any
    nrmse: Any

    def validated(self) -> "EstimatorMetric":
        if self.estimator not in {"RIDGE_PIPELINE", "EXTRA_TREES"}:
            raise IntegrityError(f"unknown estimator: {self.estimator}")
        _finite_float64(self.rmse, f"{self.estimator} RMSE")
        _finite_float64(self.r2, f"{self.estimator} R2")
        _finite_float64(self.nrmse, f"{self.estimator} NRMSE")
        return self


@dataclasses.dataclass(frozen=True, slots=True)
class EstimatorSelection:
    independent: EstimatorMetric
    parent_witnesses: tuple[EstimatorMetric, ...]
    parent_redundancy_boolean: bool


def _equal(value: Any, threshold: float) -> bool:
    np = _np()
    return bool(_finite_float64(value, "threshold value") == np.float64(threshold))


def _boundary_stop(label: str, value: Any, threshold: float) -> None:
    if _equal(value, threshold):
        raise GlobalAmbiguity(
            "AMBIGUOUS_THRESHOLD_EQUALITY:"
            f"{label}:{canonical_float_hex(value)}"
        )


def select_estimators(
    ridge: EstimatorMetric, extra_trees: EstimatorMetric
) -> EstimatorSelection:
    """Keep the independent RMSE selector separate from the parent R2 gate."""

    np = _np()
    ridge.validated()
    extra_trees.validated()
    ridge_rmse = np.float64(ridge.rmse)
    extra_rmse = np.float64(extra_trees.rmse)
    if bool(ridge_rmse == extra_rmse):
        raise GlobalAmbiguity("AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE")
    independent = ridge if bool(ridge_rmse < extra_rmse) else extra_trees

    ridge_r2 = np.float64(ridge.r2)
    extra_r2 = np.float64(extra_trees.r2)
    maximum = np.float64(max(ridge_r2, extra_r2))
    parent = tuple(
        metric
        for metric in (ridge, extra_trees)
        if bool(np.float64(metric.r2) == maximum)
    )
    for metric in parent:
        _boundary_stop(
            f"PARENT_R2_{metric.estimator}",
            metric.r2,
            PARENT_R2_CUTOFF,
        )
        _boundary_stop(
            f"PARENT_NRMSE_{metric.estimator}",
            metric.nrmse,
            PARENT_NRMSE_CUTOFF,
        )
    booleans = tuple(
        bool(
            np.float64(metric.r2) > np.float64(PARENT_R2_CUTOFF)
            and np.float64(metric.nrmse) < np.float64(PARENT_NRMSE_CUTOFF)
        )
        for metric in parent
    )
    if len(set(booleans)) != 1:
        raise GlobalAmbiguity("AMBIGUOUS_PARENT_ESTIMATOR_TIE")
    return EstimatorSelection(independent, parent, booleans[0])


@dataclasses.dataclass(frozen=True, slots=True)
class AffineScreen:
    predictor: str
    pearson: Any | None
    spearman: Any | None
    affine_oof_nrmse: Any | None
    valid: bool


def qualifying_affine_predictors(
    screens: Sequence[AffineScreen],
) -> tuple[str, ...]:
    np = _np()
    by_name = {screen.predictor: screen for screen in screens}
    if len(by_name) != len(screens):
        raise IntegrityError("duplicate affine screen predictor")
    unknown = set(by_name).difference(PREDICTORS)
    if unknown:
        raise IntegrityError(f"unknown affine predictors: {sorted(unknown)!r}")
    hits: list[str] = []
    for predictor in PREDICTORS:
        screen = by_name.get(predictor)
        if screen is None or not screen.valid:
            continue
        if (
            screen.pearson is None
            or screen.spearman is None
            or screen.affine_oof_nrmse is None
        ):
            raise IntegrityError("valid affine screen contains null metrics")
        absolute_pearson = np.float64(abs(np.float64(screen.pearson)))
        absolute_spearman = np.float64(abs(np.float64(screen.spearman)))
        affine_nrmse = _finite_float64(
            screen.affine_oof_nrmse, "affine OOF NRMSE"
        )
        _boundary_stop(
            f"AFFINE_ABS_PEARSON_{predictor}",
            absolute_pearson,
            AFFINE_CORRELATION_CUTOFF,
        )
        _boundary_stop(
            f"AFFINE_ABS_SPEARMAN_{predictor}",
            absolute_spearman,
            AFFINE_CORRELATION_CUTOFF,
        )
        _boundary_stop(
            f"AFFINE_NRMSE_{predictor}",
            affine_nrmse,
            AFFINE_NRMSE_CUTOFF,
        )
        if bool(
            absolute_pearson > np.float64(AFFINE_CORRELATION_CUTOFF)
            and absolute_spearman > np.float64(AFFINE_CORRELATION_CUTOFF)
            and affine_nrmse < np.float64(AFFINE_NRMSE_CUTOFF)
        ):
            hits.append(predictor)
    return tuple(hits)


@dataclasses.dataclass(frozen=True, slots=True)
class ComponentDecision:
    classification: str
    counts_as_nonredundant: bool
    independent_estimator: str
    parent_reference_estimators: tuple[str, ...]
    affine_hits: tuple[str, ...]


def classify_component(
    ridge: EstimatorMetric,
    extra_trees: EstimatorMetric,
    affine_screens: Sequence[AffineScreen],
    *,
    cells_pass: bool,
    distribution_pass: bool,
) -> ComponentDecision:
    """Apply the frozen component precedence using strict interiors only."""

    np = _np()
    selection = select_estimators(ridge, extra_trees)
    independent = selection.independent
    for label, value, threshold in (
        ("INDEPENDENT_R2_DUPLICATE", independent.r2, INDEPENDENT_DUPLICATE_R2_CUTOFF),
        (
            "INDEPENDENT_NRMSE_DUPLICATE",
            independent.nrmse,
            INDEPENDENT_DUPLICATE_NRMSE_CUTOFF,
        ),
        ("INDEPENDENT_R2_MARGIN", independent.r2, INDEPENDENT_MARGIN_R2_CUTOFF),
        (
            "INDEPENDENT_NRMSE_MARGIN",
            independent.nrmse,
            INDEPENDENT_MARGIN_NRMSE_CUTOFF,
        ),
    ):
        _boundary_stop(label, value, threshold)
    affine_hits = qualifying_affine_predictors(affine_screens)
    parents = tuple(metric.estimator for metric in selection.parent_witnesses)
    base = {
        "independent_estimator": independent.estimator,
        "parent_reference_estimators": parents,
        "affine_hits": affine_hits,
    }
    if affine_hits:
        return ComponentDecision(
            "CLEAR_REDUNDANT_AFFINE", False, **base
        )
    if selection.parent_redundancy_boolean:
        return ComponentDecision(
            "CLEAR_REDUNDANT_PARENT_OOF", False, **base
        )
    if bool(
        np.float64(independent.r2)
        > np.float64(INDEPENDENT_DUPLICATE_R2_CUTOFF)
        and np.float64(independent.nrmse)
        < np.float64(INDEPENDENT_DUPLICATE_NRMSE_CUTOFF)
    ):
        return ComponentDecision(
            "CLEAR_REDUNDANT_INDEPENDENT_EXACT_DUPLICATE", False, **base
        )
    margin = bool(
        np.float64(independent.r2)
        < np.float64(INDEPENDENT_MARGIN_R2_CUTOFF)
        and np.float64(independent.nrmse)
        > np.float64(INDEPENDENT_MARGIN_NRMSE_CUTOFF)
    )
    if margin and cells_pass and distribution_pass:
        return ComponentDecision(
            "PASS_NONREDUNDANT_MARGIN", True, **base
        )
    if margin:
        return ComponentDecision("CLEAR_UNSTABLE_FAILURE", False, **base)
    raise GlobalAmbiguity("AMBIGUOUS_GRAY_ZONE")


def require_exact_columns(frame: Any, expected: Sequence[str], name: str) -> None:
    observed = tuple(str(column) for column in frame.columns)
    frozen = tuple(expected)
    if observed != frozen:
        raise IntegrityError(
            f"{name} columns differ: expected={frozen!r} observed={observed!r}"
        )


def normalize_utc_int64_ns(values: Any) -> Any:
    import pandas as pd

    converted = pd.to_datetime(values, utc=True, errors="raise")
    if bool(converted.isna().any()):
        raise IntegrityError("normalized UTC key contains null")
    # pandas 3 defaults parsed datetimes to microsecond resolution.  An
    # unqualified astype("int64") would therefore silently return epoch-us,
    # which Arrow would later interpret as epoch-ns.  Freeze the unit first.
    return pd.DatetimeIndex(converted).as_unit("ns").asi8


def _normalized_key_tuples(frame: Any, key: Sequence[str]) -> tuple[tuple[Any, ...], ...]:
    import pandas as pd

    if tuple(key) not in {SITE_KEY, GROUP_KEY}:
        raise IntegrityError(f"unsupported frozen join key: {tuple(key)!r}")
    missing = set(key).difference(frame.columns)
    if missing:
        raise IntegrityError(f"join key columns missing: {sorted(missing)!r}")
    work = frame.loc[:, list(key)].copy()
    work[key[0]] = normalize_utc_int64_ns(work[key[0]])
    if bool(work.duplicated(list(key), keep=False).any()):
        raise IntegrityError(f"duplicate join keys for {tuple(key)!r}")
    tuples = [
        tuple(row)
        for row in work.sort_values(list(key), kind="mergesort").itertuples(
            index=False, name=None
        )
    ]
    if any(any(pd.isna(item) for item in row) for row in tuples):
        raise IntegrityError("join key contains null")
    return tuple(tuples)


def exact_one_to_one_join(
    external: Any,
    predictor: Any,
    *,
    key: Sequence[str],
    expected_rows: int,
) -> Any:
    """Validate exact key sets before a one-to-one, non-sorting merge."""

    left_keys = _normalized_key_tuples(external, key)
    right_keys = _normalized_key_tuples(predictor, key)
    if left_keys != right_keys:
        raise IntegrityError("external and predictor exact key sets differ")
    if len(left_keys) != expected_rows:
        raise IntegrityError(
            f"join key count {len(left_keys)} differs from {expected_rows}"
        )
    left = external.copy()
    right = predictor.copy()
    left[key[0]] = normalize_utc_int64_ns(left[key[0]])
    right[key[0]] = normalize_utc_int64_ns(right[key[0]])
    joined = left.merge(
        right,
        on=list(key),
        how="inner",
        sort=False,
        validate="one_to_one",
        indicator=True,
        suffixes=("__external", "__predictor"),
    )
    if len(joined) != expected_rows or len(joined) != len(left):
        raise IntegrityError("one-to-one merge changed the row count")
    if not bool((joined["_merge"] == "both").all()):
        raise IntegrityError("one-to-one merge contains an unmatched key")
    return joined.drop(columns=["_merge"])


def assign_fold_ordinals(operating_days: Any) -> Any:
    import pandas as pd

    values = pd.DatetimeIndex(
        pd.to_datetime(operating_days, errors="raise")
    ).date
    return _np().asarray(
        [fold_ordinal_for_operating_day(value) for value in values],
        dtype=_np().int8,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CellStrength:
    strict_not_exact_pass: bool
    weak_floor_pass: bool | None
    r2_boundary_equal: bool = False
    nrmse_boundary_equal: bool = False
    weak_floor_boundary_equal: bool = False


def evaluate_cell_strength(
    metric: MetricResult,
    *,
    weak_floor_applicable: bool,
    stop_on_boundary: bool = True,
) -> CellStrength:
    np = _np()
    metric.require_valid()
    if metric.rows < MINIMUM_CELL_ROWS:
        raise IntegrityError("valid decision cell has fewer than 48 rows")
    assert metric.r2 is not None and metric.nrmse is not None
    r2_equal = _equal(metric.r2, CELL_R2_CUTOFF)
    nrmse_equal = _equal(metric.nrmse, CELL_NRMSE_CUTOFF)
    weak_equal = bool(
        weak_floor_applicable
        and _equal(metric.nrmse, CELL_WEAK_FLOOR_NRMSE_CUTOFF)
    )
    if stop_on_boundary:
        _boundary_stop("CELL_R2", metric.r2, CELL_R2_CUTOFF)
        _boundary_stop("CELL_NRMSE", metric.nrmse, CELL_NRMSE_CUTOFF)
    if weak_floor_applicable and stop_on_boundary:
        _boundary_stop(
            "CELL_WEAK_FLOOR_NRMSE",
            metric.nrmse,
            CELL_WEAK_FLOOR_NRMSE_CUTOFF,
        )
    strict = bool(
        np.float64(metric.r2) < np.float64(CELL_R2_CUTOFF)
        or np.float64(metric.nrmse) > np.float64(CELL_NRMSE_CUTOFF)
    )
    weak = (
        bool(np.float64(metric.nrmse) > np.float64(CELL_WEAK_FLOOR_NRMSE_CUTOFF))
        if weak_floor_applicable
        else None
    )
    return CellStrength(strict, weak, r2_equal, nrmse_equal, weak_equal)


@dataclasses.dataclass(frozen=True, slots=True)
class StabilityGate:
    year_pass: bool
    season_pass: bool
    forecast_hour_pass: bool
    group_pass: bool
    site_pass: bool

    @property
    def all_pass(self) -> bool:
        return all(
            (
                self.year_pass,
                self.season_pass,
                self.forecast_hour_pass,
                self.group_pass,
                self.site_pass,
            )
        )


def evaluate_stability_gate(
    *,
    year_metrics: Sequence[MetricResult],
    season_metrics: Sequence[MetricResult],
    forecast_hour_metrics: Sequence[MetricResult],
    group_metrics: Mapping[str, MetricResult],
    site_metrics: Mapping[str, MetricResult],
    site_to_group: Mapping[str, str],
) -> StabilityGate:
    if len(year_metrics) != 2:
        raise IntegrityError("year metric count must be exactly 2")
    if len(season_metrics) != 4:
        raise IntegrityError("season metric count must be exactly 4")
    if len(forecast_hour_metrics) != 3:
        raise IntegrityError("forecast-hour metric count must be exactly 3")
    expected_groups = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
    if tuple(group_metrics) != expected_groups:
        raise IntegrityError("group metric order or labels differ")
    if len(site_metrics) != 17 or set(site_metrics) != set(site_to_group):
        raise IntegrityError("site metric or mapping coverage differs from exact 17")

    year = [evaluate_cell_strength(metric, weak_floor_applicable=False) for metric in year_metrics]
    year_pass = all(item.strict_not_exact_pass for item in year)

    seasons = [
        evaluate_cell_strength(metric, weak_floor_applicable=True)
        for metric in season_metrics
    ]
    season_strict_count = sum(item.strict_not_exact_pass for item in seasons)
    if season_strict_count == 4:
        season_pass = True
    elif season_strict_count == 3:
        remaining = next(
            item for item in seasons if not item.strict_not_exact_pass
        )
        season_pass = bool(remaining.weak_floor_pass)
    else:
        season_pass = False

    forecast = [
        evaluate_cell_strength(metric, weak_floor_applicable=True)
        for metric in forecast_hour_metrics
    ]
    forecast_pass = all(
        item.strict_not_exact_pass and bool(item.weak_floor_pass)
        for item in forecast
    )

    groups = [
        evaluate_cell_strength(group_metrics[label], weak_floor_applicable=False)
        for label in expected_groups
    ]
    group_pass = all(item.strict_not_exact_pass for item in groups)

    required = {"kpx_group_1": 5, "kpx_group_2": 5, "kpx_group_3": 4}
    totals = {"kpx_group_1": 6, "kpx_group_2": 6, "kpx_group_3": 5}
    observed_totals = {label: 0 for label in expected_groups}
    observed_pass = {label: 0 for label in expected_groups}
    for site, metric in site_metrics.items():
        group = site_to_group[site]
        if group not in observed_totals:
            raise IntegrityError(f"site maps to unknown group: {group}")
        observed_totals[group] += 1
        strength = evaluate_cell_strength(metric, weak_floor_applicable=False)
        observed_pass[group] += int(strength.strict_not_exact_pass)
    if observed_totals != totals:
        raise IntegrityError(
            f"site group cardinalities differ: {observed_totals!r}"
        )
    site_pass = all(
        observed_pass[group] >= required[group] for group in expected_groups
    )
    return StabilityGate(
        year_pass=year_pass,
        season_pass=season_pass,
        forecast_hour_pass=forecast_pass,
        group_pass=group_pass,
        site_pass=site_pass,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class MonthlyIQRResult:
    valid: bool
    invalid_reason: str | None
    iqr_2022: Any
    iqr_2023: Any
    ratio: Any | None
    passes: bool
    boundary_equal: bool = False


def monthly_iqr_stability(
    actual_2022: Any,
    actual_2023: Any,
    *,
    stop_on_boundary: bool = True,
) -> MonthlyIQRResult:
    np = _np()
    y22 = _float64_vector(actual_2022, "monthly actual 2022")
    y23 = _float64_vector(actual_2023, "monthly actual 2023")
    q22 = np.quantile(
        y22, np.asarray([0.25, 0.75], dtype=np.float64), method="linear"
    )
    q23 = np.quantile(
        y23, np.asarray([0.25, 0.75], dtype=np.float64), method="linear"
    )
    iqr22 = np.float64(q22[1] - q22[0])
    iqr23 = np.float64(q23[1] - q23[0])
    bad22 = bool(iqr22 <= np.float64(SST_GUARD))
    bad23 = bool(iqr23 <= np.float64(SST_GUARD))
    if bad22 or bad23:
        if bad22 and bad23:
            reason = "UNUSABLE_MONTHLY_IQR_BOTH_YEARS_LTE_1E_MINUS_12"
        elif bad22:
            reason = "UNUSABLE_MONTHLY_IQR_2022_LTE_1E_MINUS_12"
        else:
            reason = "UNUSABLE_MONTHLY_IQR_2023_LTE_1E_MINUS_12"
        return MonthlyIQRResult(False, reason, iqr22, iqr23, None, False, False)
    ratio = np.float64(iqr23 / iqr22)
    boundary = _equal(ratio, IQR_RATIO_MIN) or _equal(ratio, IQR_RATIO_MAX)
    if stop_on_boundary:
        _boundary_stop("MONTHLY_IQR_RATIO_MIN", ratio, IQR_RATIO_MIN)
        _boundary_stop("MONTHLY_IQR_RATIO_MAX", ratio, IQR_RATIO_MAX)
    passes = bool(
        ratio > np.float64(IQR_RATIO_MIN)
        and ratio < np.float64(IQR_RATIO_MAX)
    )
    return MonthlyIQRResult(True, None, iqr22, iqr23, ratio, passes, boundary)


@dataclasses.dataclass(frozen=True, slots=True)
class PSIResult:
    valid: bool
    invalid_reason: str | None
    internal_edges: Any
    final_bin_count: int
    reference_counts: Any
    comparison_counts: Any
    reference_probabilities: Any
    comparison_probabilities: Any
    psi: Any | None
    passes: bool
    boundary_equal: bool = False


def year_psi(
    reference_2022: Any,
    comparison_2023: Any,
    *,
    stop_on_boundary: bool = True,
) -> PSIResult:
    np = _np()
    reference = _float64_vector(reference_2022, "PSI reference 2022")
    comparison = _float64_vector(comparison_2023, "PSI comparison 2023")
    quantiles = np.arange(1, 10, dtype=np.float64) / np.float64(10.0)
    raw_edges = np.quantile(reference, quantiles, method="linear")
    edges = np.unique(np.ascontiguousarray(raw_edges, dtype=np.float64))
    k = int(edges.size + 1)
    reference_bins = np.searchsorted(edges, reference, side="right")
    comparison_bins = np.searchsorted(edges, comparison, side="right")
    reference_counts = np.bincount(reference_bins, minlength=k).astype(
        np.int64, copy=False
    )
    comparison_counts = np.bincount(comparison_bins, minlength=k).astype(
        np.int64, copy=False
    )
    if k < 2:
        empty = np.empty(0, dtype=np.float64)
        return PSIResult(
            False,
            "UNUSABLE_PSI_FINAL_BIN_COUNT_LT_2",
            edges,
            k,
            reference_counts,
            comparison_counts,
            empty,
            empty,
            None,
            False,
            False,
        )
    alpha = np.float64(0.5)
    reference_probabilities = np.ascontiguousarray(
        (reference_counts.astype(np.float64) + alpha)
        / (np.float64(reference.size) + alpha * np.float64(k)),
        dtype=np.float64,
    )
    comparison_probabilities = np.ascontiguousarray(
        (comparison_counts.astype(np.float64) + alpha)
        / (np.float64(comparison.size) + alpha * np.float64(k)),
        dtype=np.float64,
    )
    terms = np.ascontiguousarray(
        (comparison_probabilities - reference_probabilities)
        * np.log(comparison_probabilities / reference_probabilities),
        dtype=np.float64,
    )
    psi = np.float64(np.sum(terms, dtype=np.float64))
    _finite_float64(psi, "PSI")
    boundary = _equal(psi, PSI_MAX)
    if stop_on_boundary:
        _boundary_stop("YEAR_PSI", psi, PSI_MAX)
    return PSIResult(
        True,
        None,
        edges,
        k,
        reference_counts,
        comparison_counts,
        reference_probabilities,
        comparison_probabilities,
        psi,
        bool(psi < np.float64(PSI_MAX)),
        boundary,
    )


DERIVED_DIAGNOSTICS: Final[tuple[str, ...]] = (
    "WS_925",
    "WS_950",
    "WS_975",
    "WS_1000",
    "ENDPOINT_DU_925_MINUS_1000",
    "ENDPOINT_DV_925_MINUS_1000",
    "ENDPOINT_VECTOR_SHEAR_MAG",
    "ADJACENT_DU_975_MINUS_1000",
    "ADJACENT_DV_975_MINUS_1000",
    "ADJACENT_SHEAR_MAG_975_MINUS_1000",
    "ADJACENT_DU_950_MINUS_975",
    "ADJACENT_DV_950_MINUS_975",
    "ADJACENT_SHEAR_MAG_950_MINUS_975",
    "ADJACENT_DU_925_MINUS_950",
    "ADJACENT_DV_925_MINUS_950",
    "ADJACENT_SHEAR_MAG_925_MINUS_950",
    "ENDPOINT_DIRECTION_COS",
    "ENDPOINT_DIRECTION_SIN",
)
ENDPOINT_VETO_DIAGNOSTICS: Final[tuple[str, ...]] = (
    "ENDPOINT_DU_925_MINUS_1000",
    "ENDPOINT_DV_925_MINUS_1000",
    "ENDPOINT_VECTOR_SHEAR_MAG",
)
CALM_THRESHOLD_MPS: Final[float] = 1e-6


def derive_wind_diagnostics(
    raw_components: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the exact 18 wind diagnostics from site or group U/V arrays."""

    np = _np()
    required = {
        f"{component}GRD_{level}mb"
        for level in (925, 950, 975, 1000)
        for component in ("U", "V")
    }
    if set(raw_components) != required:
        raise IntegrityError(
            "derived wind input component set differs from exact eight"
        )
    arrays = {
        name: _float64_vector(values, f"derived input {name}")
        for name, values in raw_components.items()
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise IntegrityError("derived wind component shapes differ")

    u = {level: arrays[f"UGRD_{level}mb"] for level in (925, 950, 975, 1000)}
    v = {level: arrays[f"VGRD_{level}mb"] for level in (925, 950, 975, 1000)}
    result: dict[str, Any] = {}
    speed: dict[int, Any] = {}
    for level in (925, 950, 975, 1000):
        speed[level] = np.ascontiguousarray(np.hypot(u[level], v[level]), dtype=np.float64)
        result[f"WS_{level}"] = speed[level]

    endpoint_du = np.ascontiguousarray(u[925] - u[1000], dtype=np.float64)
    endpoint_dv = np.ascontiguousarray(v[925] - v[1000], dtype=np.float64)
    result["ENDPOINT_DU_925_MINUS_1000"] = endpoint_du
    result["ENDPOINT_DV_925_MINUS_1000"] = endpoint_dv
    result["ENDPOINT_VECTOR_SHEAR_MAG"] = np.ascontiguousarray(
        np.hypot(endpoint_du, endpoint_dv), dtype=np.float64
    )

    for lower, upper in ((1000, 975), (975, 950), (950, 925)):
        du = np.ascontiguousarray(u[upper] - u[lower], dtype=np.float64)
        dv = np.ascontiguousarray(v[upper] - v[lower], dtype=np.float64)
        result[f"ADJACENT_DU_{upper}_MINUS_{lower}"] = du
        result[f"ADJACENT_DV_{upper}_MINUS_{lower}"] = dv
        result[f"ADJACENT_SHEAR_MAG_{upper}_MINUS_{lower}"] = (
            np.ascontiguousarray(np.hypot(du, dv), dtype=np.float64)
        )

    product = np.ascontiguousarray(speed[1000] * speed[925], dtype=np.float64)
    denominator = np.maximum(product, np.float64(CALM_THRESHOLD_MPS))
    cosine = np.clip(
        (u[1000] * u[925] + v[1000] * v[925]) / denominator,
        np.float64(-1.0),
        np.float64(1.0),
    )
    sine = np.clip(
        (u[1000] * v[925] - v[1000] * u[925]) / denominator,
        np.float64(-1.0),
        np.float64(1.0),
    )
    calm = (speed[1000] < np.float64(CALM_THRESHOLD_MPS)) | (
        speed[925] < np.float64(CALM_THRESHOLD_MPS)
    )
    cosine = np.ascontiguousarray(cosine, dtype=np.float64)
    sine = np.ascontiguousarray(sine, dtype=np.float64)
    cosine[calm] = np.float64(0.0)
    sine[calm] = np.float64(0.0)
    result["ENDPOINT_DIRECTION_COS"] = cosine
    result["ENDPOINT_DIRECTION_SIN"] = sine

    if tuple(result) != DERIVED_DIAGNOSTICS:
        raise AssertionError("derived diagnostic insertion order drift")
    for name, array in result.items():
        if not bool(np.isfinite(array).all()):
            raise IntegrityError(f"derived diagnostic is non-finite: {name}")
    return result


def capacity_aggregate_raw_uv(
    frame: Any,
    *,
    value_columns: Sequence[str] = WIND_RESPONSES,
) -> Any:
    """Capacity-aggregate raw U/V first; derived values are computed afterward."""

    import pandas as pd

    required = {"valid_time_utc", "group", "capacity_mw", *value_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise IntegrityError(
            f"capacity aggregation columns missing: {sorted(missing)!r}"
        )
    work = frame.loc[:, ["valid_time_utc", "group", "capacity_mw", *value_columns]].copy()
    work["valid_time_utc"] = normalize_utc_int64_ns(work["valid_time_utc"])
    capacity = work["capacity_mw"].to_numpy(dtype="float64", copy=True)
    if not bool(_np().isfinite(capacity).all()) or bool((capacity <= 0.0).any()):
        raise IntegrityError("capacity weights must be finite and positive")
    weighted_names: list[str] = []
    for column in value_columns:
        values = work[column].to_numpy(dtype="float64", copy=True)
        if not bool(_np().isfinite(values).all()):
            raise IntegrityError(f"group aggregation input is non-finite: {column}")
        weighted = f"__weighted__{column}"
        work[weighted] = values * capacity
        weighted_names.append(weighted)
    grouped = (
        work.groupby(["valid_time_utc", "group"], sort=True, observed=True)
        .agg({"capacity_mw": "sum", **{name: "sum" for name in weighted_names}})
        .reset_index()
    )
    for column, weighted in zip(value_columns, weighted_names, strict=True):
        grouped[column] = (
            grouped[weighted].to_numpy(dtype="float64", copy=False)
            / grouped["capacity_mw"].to_numpy(dtype="float64", copy=False)
        )
    return grouped.loc[
        :, ["valid_time_utc", "group", "capacity_mw", *value_columns]
    ].sort_values(["valid_time_utc", "group"], kind="mergesort", ignore_index=True)


def make_ridge_pipeline() -> Any:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            (
                "standard_scaler",
                StandardScaler(copy=True, with_mean=True, with_std=True),
            ),
            (
                "ridge",
                Ridge(
                    alpha=1.0,
                    copy_X=True,
                    fit_intercept=True,
                    max_iter=None,
                    positive=False,
                    random_state=None,
                    solver="cholesky",
                    tol=0.0001,
                ),
            ),
        ],
        memory=None,
        transform_input=None,
        verbose=False,
    )


def make_extra_trees() -> Any:
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=128,
        criterion="squared_error",
        max_depth=16,
        min_samples_split=2,
        min_samples_leaf=8,
        min_weight_fraction_leaf=0.0,
        max_features=0.8,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=7,
        random_state=260810,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    )


def manual_ridge_pipeline_predict(pipeline: Any, heldout_x: Any) -> Any:
    """Predict in a fixed float64 order from a fitted scaler and Ridge."""

    np = _np()
    if tuple(pipeline.named_steps) != ("standard_scaler", "ridge"):
        raise IntegrityError("Ridge pipeline step order differs")
    scaler = pipeline.named_steps["standard_scaler"]
    ridge = pipeline.named_steps["ridge"]
    x = np.ascontiguousarray(heldout_x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(PREDICTORS):
        raise IntegrityError("Ridge heldout matrix shape differs")
    mean = np.ascontiguousarray(scaler.mean_, dtype=np.float64)
    scale = np.ascontiguousarray(scaler.scale_, dtype=np.float64)
    coefficient = np.ascontiguousarray(ridge.coef_, dtype=np.float64)
    intercept = np.float64(ridge.intercept_)
    if mean.shape != (len(PREDICTORS),) or scale.shape != mean.shape:
        raise IntegrityError("fitted StandardScaler shape differs")
    if coefficient.shape != mean.shape:
        raise IntegrityError("fitted Ridge coefficient shape differs")
    transformed = np.ascontiguousarray((x - mean) / scale, dtype=np.float64)
    prediction = np.ascontiguousarray(
        np.sum(transformed * coefficient, axis=1, dtype=np.float64) + intercept,
        dtype=np.float64,
    )
    repeated = np.ascontiguousarray(
        np.sum(transformed * coefficient, axis=1, dtype=np.float64) + intercept,
        dtype=np.float64,
    )
    if not bool(np.array_equal(prediction, repeated)):
        raise IntegrityError("Ridge repeated prediction is not array-equal")
    if not bool(np.isfinite(prediction).all()):
        raise IntegrityError("Ridge prediction is non-finite")
    return prediction


def serial_extra_trees_predict(forest: Any, heldout_x: Any) -> Any:
    """Use fitted trees in exact order with one float64 accumulator."""

    np = _np()
    estimators = tuple(forest.estimators_)
    if len(estimators) != 128:
        raise IntegrityError("ExtraTrees fitted estimator count differs from 128")
    x = np.ascontiguousarray(heldout_x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(PREDICTORS):
        raise IntegrityError("ExtraTrees heldout matrix shape differs")

    def predict_once() -> Any:
        accumulator = np.zeros(x.shape[0], dtype=np.float64)
        for tree in estimators:
            prediction = np.asarray(
                tree.predict(x, check_input=False), dtype=np.float64
            )
            if prediction.shape != accumulator.shape:
                raise IntegrityError("tree prediction shape differs")
            np.add(accumulator, prediction, out=accumulator)
        accumulator /= np.float64(128.0)
        return accumulator

    first = predict_once()
    second = predict_once()
    if not bool(np.array_equal(first, second)):
        raise IntegrityError("ExtraTrees repeated prediction is not array-equal")
    if not bool(np.isfinite(first).all()):
        raise IntegrityError("ExtraTrees prediction is non-finite")
    return first


@dataclasses.dataclass(frozen=True, slots=True)
class PrimaryCrossfitResult:
    ridge_oof: Any
    extra_trees_oof: Any
    completed_ledger_records: tuple[dict[str, Any], ...]
    internal_method_calls: Mapping[str, int]


def _validate_exact_crossfit_arrays(
    predictors: Any, response: Any, fold_ordinals: Any
) -> tuple[Any, Any, Any]:
    np = _np()
    x64 = np.ascontiguousarray(predictors, dtype=np.float64)
    y = _float64_vector(response, "crossfit response")
    folds = np.ascontiguousarray(fold_ordinals, dtype=np.int8)
    if x64.shape != (EXPECTED_SITE_ROWS, len(PREDICTORS)):
        raise IntegrityError(
            "crossfit predictor shape must be exactly (19584, 35)"
        )
    if y.shape != (EXPECTED_SITE_ROWS,) or folds.shape != y.shape:
        raise IntegrityError("crossfit response or fold shape differs")
    if not bool(np.isfinite(x64).all()):
        raise IntegrityError("crossfit predictor contains non-finite values")
    expected_fold_rows = 4_896
    for fold in FOLDS:
        if int(np.count_nonzero(folds == fold.ordinal)) != expected_fold_rows:
            raise IntegrityError(
                f"heldout row count differs for fold {fold.name}"
            )
    if not bool(np.isin(folds, np.array([1, 2, 3, 4], dtype=np.int8)).all()):
        raise IntegrityError("crossfit fold ordinal is outside 1..4")
    return x64, y, folds


def crossfit_primary_response(
    predictors: Any,
    response: Any,
    fold_ordinals: Any,
    *,
    response_name: str,
    authorization: PredataAuthorization,
) -> PrimaryCrossfitResult:
    """Execute the exact eight primary units for one nonconstant response."""

    from joblib import parallel_backend
    from threadpoolctl import threadpool_limits

    np = _np()
    authorization.assert_arrow_handoff()
    if response_name not in RESPONSES:
        raise IntegrityError(f"unknown response: {response_name}")
    x64, y, folds = _validate_exact_crossfit_arrays(
        predictors, response, fold_ordinals
    )
    if premodel_response_skip_reason(y) is not None:
        raise IntegrityError("skipped response must not enter primary fitting")
    response_ordinal = RESPONSES.index(response_name) + 1
    ridge_oof = np.full(y.shape, np.nan, dtype=np.float64)
    trees_oof = np.full(y.shape, np.nan, dtype=np.float64)
    ridge_fill = np.zeros(y.shape, dtype=np.int8)
    trees_fill = np.zeros(y.shape, dtype=np.int8)
    records: list[dict[str, Any]] = []
    internal = {
        "sklearn.pipeline.Pipeline.fit": 0,
        "sklearn.preprocessing.StandardScaler.fit": 0,
        "sklearn.linear_model.Ridge.fit": 0,
        "sklearn.ensemble.ExtraTreesRegressor.fit": 0,
    }
    for fold in FOLDS:
        held = folds == fold.ordinal
        train = ~held
        if int(np.count_nonzero(train)) != 14_688:
            raise IntegrityError("training row count differs from 14688")
        x_train_ridge = np.ascontiguousarray(x64[train], dtype=np.float64)
        x_held_ridge = np.ascontiguousarray(x64[held], dtype=np.float64)
        y_train = np.ascontiguousarray(y[train], dtype=np.float64)
        ridge_fit_label = f"{response_name}/{fold.name}/RIDGE_PIPELINE/FIT"
        with threadpool_limits(limits=1):
            authorization.capture_threadpool(
                phase="INSIDE_CONTEXT", label=ridge_fit_label
            )
            pipeline = make_ridge_pipeline()
            pipeline.fit(x_train_ridge, y_train)
        authorization.capture_threadpool(
            phase="AFTER_CONTEXT", label=ridge_fit_label
        )
        ridge_prediction_label = (
            f"{response_name}/{fold.name}/RIDGE_PIPELINE/PREDICT"
        )
        with threadpool_limits(limits=1):
            authorization.capture_threadpool(
                phase="INSIDE_CONTEXT", label=ridge_prediction_label
            )
            ridge_prediction = manual_ridge_pipeline_predict(
                pipeline, x_held_ridge
            )
        authorization.capture_threadpool(
            phase="AFTER_CONTEXT", label=ridge_prediction_label
        )
        internal["sklearn.pipeline.Pipeline.fit"] += 1
        internal["sklearn.preprocessing.StandardScaler.fit"] += 1
        internal["sklearn.linear_model.Ridge.fit"] += 1
        ridge_oof[held] = ridge_prediction
        ridge_fill[held] += np.int8(1)
        ridge_unit = PlannedUnit(
            f"PMU__{response_ordinal:02d}__{fold.ordinal}__1",
            (response_ordinal - 1) * 8 + (fold.ordinal - 1) * 2 + 1,
            "PRIMARY_MODEL",
            response_name,
            fold.name,
            "RIDGE_PIPELINE",
            None,
        )
        records.append(
            {
                "planned_unit_slot_id": ridge_unit.planned_unit_slot_id,
                "decision_fit_ordinal": ridge_unit.decision_fit_ordinal,
                "unit_kind": "PRIMARY_MODEL",
                "response": response_name,
                "fold": fold.name,
                "estimator_unit": "RIDGE_PIPELINE",
                "training_rows": 14_688,
                "training_columns": 35,
                "heldout_rows": 4_896,
                "input_dtype": "float64_C_CONTIGUOUS",
                "response_dtype": "float64",
                "unit_status": UNIT_COMPLETED,
                "skip_reason": None,
                "pipeline_fit_calls": 1,
                "standard_scaler_fit_calls": 1,
                "ridge_fit_calls": 1,
                "extra_trees_fit_calls": 0,
                "predict_calls": 1,
                "random_state": None,
                "fit_completed": True,
            }
        )

        x_train_trees = np.ascontiguousarray(x64[train], dtype=np.float32)
        x_held_trees = np.ascontiguousarray(x64[held], dtype=np.float32)
        trees_fit_label = f"{response_name}/{fold.name}/EXTRA_TREES/FIT"
        with threadpool_limits(limits=1), parallel_backend(
            "threading", n_jobs=7
        ):
            authorization.capture_threadpool(
                phase="INSIDE_CONTEXT", label=trees_fit_label
            )
            forest = make_extra_trees()
            forest.fit(x_train_trees, y_train)
        authorization.capture_threadpool(
            phase="AFTER_CONTEXT", label=trees_fit_label
        )
        if any(int(tree.max_features_) != 28 for tree in forest.estimators_):
            raise IntegrityError("ExtraTrees resolved max_features differs from 28")
        trees_prediction_label = (
            f"{response_name}/{fold.name}/EXTRA_TREES/PREDICT"
        )
        with threadpool_limits(limits=1):
            authorization.capture_threadpool(
                phase="INSIDE_CONTEXT", label=trees_prediction_label
            )
            trees_prediction = serial_extra_trees_predict(
                forest, x_held_trees
            )
        authorization.capture_threadpool(
            phase="AFTER_CONTEXT", label=trees_prediction_label
        )
        internal["sklearn.ensemble.ExtraTreesRegressor.fit"] += 1
        trees_oof[held] = trees_prediction
        trees_fill[held] += np.int8(1)
        records.append(
            {
                "planned_unit_slot_id": (
                    f"PMU__{response_ordinal:02d}__{fold.ordinal}__2"
                ),
                "decision_fit_ordinal": (
                    (response_ordinal - 1) * 8 + (fold.ordinal - 1) * 2 + 2
                ),
                "unit_kind": "PRIMARY_MODEL",
                "response": response_name,
                "fold": fold.name,
                "estimator_unit": "EXTRA_TREES",
                "training_rows": 14_688,
                "training_columns": 35,
                "heldout_rows": 4_896,
                "input_dtype": "float32_C_CONTIGUOUS",
                "response_dtype": "float64",
                "unit_status": UNIT_COMPLETED,
                "skip_reason": None,
                "pipeline_fit_calls": 0,
                "standard_scaler_fit_calls": 0,
                "ridge_fit_calls": 0,
                "extra_trees_fit_calls": 1,
                "predict_calls": 1,
                "random_state": 260810,
                "fit_completed": True,
            }
        )
    if not bool(np.array_equal(ridge_fill, np.ones_like(ridge_fill))):
        raise IntegrityError("Ridge OOF fill count is not exactly one")
    if not bool(np.array_equal(trees_fill, np.ones_like(trees_fill))):
        raise IntegrityError("ExtraTrees OOF fill count is not exactly one")
    if not bool(np.isfinite(ridge_oof).all()) or not bool(
        np.isfinite(trees_oof).all()
    ):
        raise IntegrityError("primary OOF contains an unfilled or non-finite row")
    if len(records) != 8:
        raise AssertionError("per-response primary ledger count drift")
    return PrimaryCrossfitResult(
        ridge_oof=ridge_oof,
        extra_trees_oof=trees_oof,
        completed_ledger_records=tuple(records),
        internal_method_calls=internal,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class AffineCrossfitResult:
    screens: tuple[AffineScreen, ...]
    predictor_oof: Mapping[str, Any]
    completed_ledger_records: tuple[dict[str, Any], ...]


def crossfit_affine_response(
    predictors: Any,
    response: Any,
    fold_ordinals: Any,
    *,
    response_name: str,
    pooled_denominator: Any,
    authorization: PredataAuthorization,
) -> AffineCrossfitResult:
    """Execute the exact 140 analytic units for one nonconstant response."""

    from threadpoolctl import threadpool_limits

    np = _np()
    authorization.assert_arrow_handoff()
    if response_name not in RESPONSES:
        raise IntegrityError(f"unknown response: {response_name}")
    x64, y, folds = _validate_exact_crossfit_arrays(
        predictors, response, fold_ordinals
    )
    response_ordinal = RESPONSES.index(response_name) + 1
    records: list[dict[str, Any]] = []
    screens: list[AffineScreen] = []
    oof_by_predictor: dict[str, Any] = {}
    affine_base = (response_ordinal - 1) * len(PREDICTORS) * len(FOLDS)
    for predictor_ordinal, predictor in enumerate(PREDICTORS, start=1):
        oof = np.full(y.shape, np.nan, dtype=np.float64)
        fill = np.zeros(y.shape, dtype=np.int8)
        for fold in FOLDS:
            held = folds == fold.ordinal
            train = ~held
            affine_label = (
                f"{response_name}/{predictor}/{fold.name}/"
                "ANALYTIC_AFFINE/FIT_PREDICT"
            )
            with threadpool_limits(limits=1):
                authorization.capture_threadpool(
                    phase="INSIDE_CONTEXT", label=affine_label
                )
                result = analytic_affine_fit_predict(
                    x64[train, predictor_ordinal - 1],
                    y[train],
                    x64[held, predictor_ordinal - 1],
                )
            authorization.capture_threadpool(
                phase="AFTER_CONTEXT", label=affine_label
            )
            oof[held] = result.prediction
            fill[held] += np.int8(1)
            affine_ordinal = (
                affine_base
                + (predictor_ordinal - 1) * len(FOLDS)
                + fold.ordinal
            )
            decision_ordinal = EXPECTED_PRIMARY_UNITS + affine_ordinal
            slope, slope_hex = persist_float(result.slope)
            intercept, intercept_hex = persist_float(result.intercept)
            sxx, sxx_hex = persist_float(result.sxx)
            records.append(
                {
                    "planned_unit_slot_id": (
                        f"AAU__{response_ordinal:02d}__"
                        f"{predictor_ordinal:02d}__{fold.ordinal}"
                    ),
                    "decision_fit_ordinal": decision_ordinal,
                    "analytic_affine_ordinal": affine_ordinal,
                    "unit_kind": "ANALYTIC_AFFINE",
                    "response": response_name,
                    "predictor": predictor,
                    "fold": fold.name,
                    "training_rows": 14_688,
                    "heldout_rows": 4_896,
                    "unit_status": UNIT_COMPLETED,
                    "skip_reason": None,
                    "sxx": sxx,
                    "sxx_float_hex": sxx_hex,
                    "constant_predictor_branch": result.constant_predictor_branch,
                    "slope": slope,
                    "slope_float_hex": slope_hex,
                    "intercept": intercept,
                    "intercept_float_hex": intercept_hex,
                    "fit_completed": True,
                }
            )
        if not bool(np.array_equal(fill, np.ones_like(fill))):
            raise IntegrityError("affine OOF fill count is not exactly one")
        if not bool(np.isfinite(oof).all()):
            raise IntegrityError("affine OOF contains non-finite values")
        pearson = pearson_correlation(x64[:, predictor_ordinal - 1], y)
        spearman = spearman_correlation(x64[:, predictor_ordinal - 1], y)
        metrics = regression_metrics(y, oof, pooled_denominator)
        if not metrics.valid or metrics.nrmse is None:
            raise IntegrityError("nonconstant affine pooled metric is unusable")
        screens.append(
            AffineScreen(
                predictor=predictor,
                pearson=pearson.value,
                spearman=spearman.value,
                affine_oof_nrmse=metrics.nrmse,
                valid=pearson.valid and spearman.valid,
            )
        )
        oof_by_predictor[predictor] = oof
    if len(records) != 140 or len(screens) != 35:
        raise AssertionError("per-response affine count drift")
    return AffineCrossfitResult(
        screens=tuple(screens),
        predictor_oof=oof_by_predictor,
        completed_ledger_records=tuple(records),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class FamilyDecision:
    status: str
    selected_family: str | None
    selected_raw_columns: tuple[str, ...]
    selected_derived_columns: tuple[str, ...]
    pbl_clear_pass: bool
    wind_clear_pass: bool


def decide_family(
    component_results: Mapping[str, ComponentDecision],
    endpoint_veto_pass: Mapping[str, bool],
) -> FamilyDecision:
    if tuple(component_results) != RESPONSES:
        raise IntegrityError("component result order or coverage differs")
    if tuple(endpoint_veto_pass) != ENDPOINT_VETO_DIAGNOSTICS:
        raise IntegrityError("endpoint veto order or coverage differs")
    pbl_pass = (
        component_results["HPBL_surface"].classification
        == "PASS_NONREDUNDANT_MARGIN"
    )
    passed_wind = [
        name
        for name in WIND_RESPONSES
        if component_results[name].classification == "PASS_NONREDUNDANT_MARGIN"
    ]
    components = {name.split("_", 1)[0] for name in passed_wind}
    levels = {int(name.split("_", 1)[1][:-2]) for name in passed_wind}
    wind_pass = bool(
        all(
            component_results[name].classification
            != "CLEAR_PHYSICAL_OR_COVERAGE_FAILURE"
            for name in WIND_RESPONSES
        )
        and len(passed_wind) >= 4
        and {"UGRD", "VGRD"}.issubset(components)
        and len(levels) >= 2
        and all(endpoint_veto_pass.values())
    )
    if wind_pass:
        return FamilyDecision(
            status=(
                "PASS_ONE_TARGET_FREE_FAMILY_SELECTED_PENDING_"
                "INDEPENDENT_POSTRUN_SEAL"
            ),
            selected_family="LOW_LEVEL_ISOBARIC_WIND_PROFILE",
            selected_raw_columns=WIND_RESPONSES,
            selected_derived_columns=DERIVED_DIAGNOSTICS,
            pbl_clear_pass=pbl_pass,
            wind_clear_pass=True,
        )
    if pbl_pass:
        return FamilyDecision(
            status=(
                "PASS_ONE_TARGET_FREE_FAMILY_SELECTED_PENDING_"
                "INDEPENDENT_POSTRUN_SEAL"
            ),
            selected_family="PBL_HEIGHT",
            selected_raw_columns=PBL_RESPONSES,
            selected_derived_columns=(),
            pbl_clear_pass=True,
            wind_clear_pass=False,
        )
    return FamilyDecision(
        status="STOP_NO_TARGET_FREE_INCREMENTAL_FAMILY",
        selected_family=None,
        selected_raw_columns=(),
        selected_derived_columns=(),
        pbl_clear_pass=False,
        wind_clear_pass=False,
    )


RUNNER_ARTIFACT_BASENAMES: Final[tuple[str, ...]] = (
    "TARGET_FREE_DUPLICATE_METRICS.csv",
    "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv",
    "TARGET_FREE_FAMILY_DECISION.json",
    "TARGET_FREE_DUPLICATE_OOF_V5.parquet",
    "TARGET_FREE_FIT_LEDGER_V5.json",
    "TARGET_FREE_DISTRIBUTION_STABILITY_V5.json",
    "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V5.parquet",
)
SEALER_ARTIFACT_BASENAMES: Final[tuple[str, ...]] = (
    "TARGET_FREE_FAMILY_LOCK.json",
    "TRACK_A_TARGET_FREE_RUN_MANIFEST.json",
)
ALIAS_BASENAMES: Final[tuple[str, ...]] = (
    "FIELD_CENSUS_LOCK.json",
    "PROVENANCE_LEDGER.parquet",
)
COMPLETE_ARTIFACT_BASENAMES: Final[tuple[str, ...]] = (
    *ALIAS_BASENAMES,
    "TARGET_FREE_DUPLICATE_METRICS.csv",
    "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv",
    "TARGET_FREE_FAMILY_DECISION.json",
    "TARGET_FREE_DUPLICATE_OOF_V5.parquet",
    "TARGET_FREE_FIT_LEDGER_V5.json",
    "TARGET_FREE_DISTRIBUTION_STABILITY_V5.json",
    "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V5.parquet",
    "TARGET_FREE_FAMILY_LOCK.json",
    "TRACK_A_TARGET_FREE_RUN_MANIFEST.json",
)

DUPLICATE_METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "record_kind",
    "response",
    "estimator",
    "predictor",
    "rows",
    "valid",
    "invalid_reason",
    "sse",
    "sse_float_hex",
    "rmse",
    "rmse_float_hex",
    "r2",
    "r2_float_hex",
    "nrmse",
    "nrmse_float_hex",
    "actual_q05",
    "actual_q05_float_hex",
    "actual_q95",
    "actual_q95_float_hex",
    "nrmse_denominator",
    "nrmse_denominator_float_hex",
    "pearson",
    "pearson_float_hex",
    "spearman",
    "spearman_float_hex",
    "affine_oof_nrmse",
    "affine_oof_nrmse_float_hex",
    "independent_selected",
    "parent_max_r2_witness",
    "parent_redundancy_boolean",
    "affine_qualifies",
    "boundary_equality_flags",
    "component_classification",
)
STABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "diagnostic_kind",
    "response_or_endpoint",
    "scope_type",
    "scope_label",
    "rows",
    "valid",
    "invalid_reason",
    "unique_count",
    "sst",
    "sst_float_hex",
    "sse",
    "sse_float_hex",
    "rmse",
    "rmse_float_hex",
    "r2",
    "r2_float_hex",
    "nrmse",
    "nrmse_float_hex",
    "pooled_nrmse_denominator",
    "pooled_nrmse_denominator_float_hex",
    "r2_boundary_equal_0_995",
    "nrmse_boundary_equal_0_02",
    "weak_floor_applicable",
    "nrmse_boundary_equal_0_01",
    "strict_not_exact_pass",
    "weak_floor_pass",
    "cell_verdict",
)
OOF_COLUMNS: Final[tuple[str, ...]] = (
    "planned_oof_slot_id",
    "response",
    "estimator",
    "fold",
    "valid_time_utc",
    "target_operating_day_kst",
    "forecast_hour",
    "site_id",
    "group",
    "capacity_mw",
    "actual",
    "oof_prediction",
    "residual",
    "slot_status",
    "invalid_reason",
    "independent_selected",
)
DERIVED_COLUMNS: Final[tuple[str, ...]] = (
    "planned_derived_slot_id",
    "diagnostic",
    "formula_id",
    "scope_type",
    "valid_time_utc",
    "target_operating_day_kst",
    "forecast_hour",
    "site_id",
    "group",
    "capacity_mw",
    "actual",
    "oof_prediction",
    "residual",
    "valid",
    "invalid_reason",
    "endpoint_veto_required",
    "source_component_estimator_binding_json",
)

FIT_LEDGER_TOP_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "artifact_type",
    "status",
    "planned_counts",
    "executed_counts",
    "internal_method_call_counts",
    "skipped_counts",
    "zero_fit_counts",
    "unit_slots",
)
FIT_LEDGER_PLANNED_COUNT_KEYS: Final[tuple[str, ...]] = (
    "primary_model_units",
    "analytic_affine_units",
    "total_decision_units",
    "fit_ledger_rows",
)
FIT_LEDGER_EXECUTED_COUNT_KEYS: Final[tuple[str, ...]] = (
    "completed_primary_model_units",
    "completed_analytic_affine_units",
    "completed_total_decision_units",
)
FIT_LEDGER_METHOD_COUNT_KEYS: Final[tuple[str, ...]] = (
    "sklearn.pipeline.Pipeline.fit",
    "sklearn.preprocessing.StandardScaler.fit",
    "sklearn.linear_model.Ridge.fit",
    "sklearn.ensemble.ExtraTreesRegressor.fit",
    "total_method_calls",
)
FIT_LEDGER_SKIPPED_COUNT_KEYS: Final[tuple[str, ...]] = (
    "skipped_primary_model_units",
    "skipped_analytic_affine_units",
    "skipped_total_decision_units",
)
FIT_LEDGER_ZERO_COUNT_KEYS: Final[tuple[str, ...]] = (
    "derived",
    "group",
    "hyperparameter_or_seed_search",
)
DISTRIBUTION_TOP_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "artifact_type",
    "status",
    "scope_order",
    "diagnostic_order",
    "endpoint_pooled_metric_records",
    "records",
)
DISTRIBUTION_SCOPE_ORDER: Final[tuple[str, ...]] = (
    "SITE_POOLED",
    "kpx_group_1",
    "kpx_group_2",
    "kpx_group_3",
)


def _derive_family_clear_passes(
    raw_results: Sequence[Mapping[str, Any]],
    endpoint_results: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    if len(raw_results) != len(RESPONSES) or len(endpoint_results) != len(
        ENDPOINT_VETO_DIAGNOSTICS
    ):
        raise IntegrityError("family evidence cardinality differs")
    by_response = {
        str(record["response"]): record for record in raw_results
    }
    if tuple(by_response) != RESPONSES:
        raise IntegrityError("family raw evidence order differs")
    passed_wind = [
        response
        for response in WIND_RESPONSES
        if by_response[response]["classification"]
        == "PASS_NONREDUNDANT_MARGIN"
    ]
    passed_components = {name.split("_", 1)[0] for name in passed_wind}
    passed_levels = {int(name.split("_", 1)[1][:-2]) for name in passed_wind}
    wind_pass = bool(
        all(
            by_response[response]["classification"]
            != "CLEAR_PHYSICAL_OR_COVERAGE_FAILURE"
            for response in WIND_RESPONSES
        )
        and len(passed_wind) >= 4
        and {"UGRD", "VGRD"}.issubset(passed_components)
        and len(passed_levels) >= 2
        and all(record["clear_pass"] is True for record in endpoint_results)
    )
    pbl_pass = bool(
        by_response["HPBL_surface"]["classification"]
        == "PASS_NONREDUNDANT_MARGIN"
    )
    return {
        "LOW_LEVEL_ISOBARIC_WIND_PROFILE": wind_pass,
        "PBL_HEIGHT": pbl_pass,
    }


def _global_ambiguity_reasons(
    raw_results: Sequence[Mapping[str, Any]],
    threshold_witnesses: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    for item in raw_results:
        if item["classification"] == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE":
            reasons.append(
                "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE:" + item["response"]
            )
    for witness in threshold_witnesses:
        if witness["boundary_equal"]:
            reasons.append(
                "AMBIGUOUS_THRESHOLD_EQUALITY:"
                + witness["cutoff_name"]
                + ":"
                + witness["scope"]
                + ":"
                + witness["diagnostic"]
                + ":"
                + witness["value_float_hex"]
            )
    for item in raw_results:
        if item["classification"] == "AMBIGUOUS_PARENT_ESTIMATOR_TIE":
            reasons.append(
                "AMBIGUOUS_PARENT_ESTIMATOR_TIE:" + item["response"]
            )
    for item in raw_results:
        if item["classification"] == "AMBIGUOUS_GRAY_ZONE":
            reasons.append("AMBIGUOUS_GRAY_ZONE:" + item["response"])
    if len(reasons) != len(set(reasons)):
        raise IntegrityError("global ambiguity reason is not unique")
    return reasons


def build_family_decision_document(
    raw_evaluations: Mapping[str, ResponseEvaluation],
    derived: DerivedEvaluation,
) -> dict[str, Any]:
    if tuple(raw_evaluations) != RESPONSES:
        raise IntegrityError("raw evaluation order or coverage differs")
    raw_results = [
        dict(raw_evaluations[response].raw_result) for response in RESPONSES
    ]
    endpoint_results = [dict(item) for item in derived.endpoint_results]
    if [item["diagnostic"] for item in endpoint_results] != list(
        ENDPOINT_VETO_DIAGNOSTICS
    ):
        raise IntegrityError("endpoint evaluation order differs")
    thresholds = [
        dict(witness)
        for response in RESPONSES
        for witness in raw_evaluations[response].threshold_witnesses
    ]
    thresholds.extend(dict(item) for item in derived.threshold_witnesses)
    thresholds.sort(key=_threshold_sort_key)
    ties = [
        dict(witness)
        for response in RESPONSES
        for witness in raw_evaluations[response].tied_witnesses
    ]
    ties.sort(
        key=lambda item: (
            RESPONSES.index(item["diagnostic"]),
            TIE_KINDS.index(item["tie_kind"]),
        )
    )
    reasons = _global_ambiguity_reasons(raw_results, thresholds)
    global_ambiguity = bool(reasons)
    endpoint_ambiguity = any(
        item["verdict"]
        in {"AMBIGUOUS_THRESHOLD_EQUALITY", "AMBIGUOUS_RAW_ESTIMATOR_TIE"}
        for item in endpoint_results
    )
    raw_ambiguity = any(
        item["classification"] in AMBIGUOUS_CLASSIFICATIONS
        for item in raw_results
    )
    if global_ambiguity is not (raw_ambiguity or endpoint_ambiguity):
        raise IntegrityError("global ambiguity reason coverage differs")
    family_clear = _derive_family_clear_passes(raw_results, endpoint_results)
    if global_ambiguity:
        status = "STOP_FAMILY_AMBIGUOUS"
        selected_family = None
        selected_raw: list[str] = []
        selected_derived: list[str] = []
    elif family_clear["LOW_LEVEL_ISOBARIC_WIND_PROFILE"]:
        status = (
            "PASS_ONE_TARGET_FREE_FAMILY_SELECTED_PENDING_"
            "INDEPENDENT_POSTRUN_SEAL"
        )
        selected_family = "LOW_LEVEL_ISOBARIC_WIND_PROFILE"
        selected_raw = list(WIND_RESPONSES)
        selected_derived = list(DERIVED_DIAGNOSTICS)
    elif family_clear["PBL_HEIGHT"]:
        status = (
            "PASS_ONE_TARGET_FREE_FAMILY_SELECTED_PENDING_"
            "INDEPENDENT_POSTRUN_SEAL"
        )
        selected_family = "PBL_HEIGHT"
        selected_raw = list(PBL_RESPONSES)
        selected_derived = []
    else:
        status = "STOP_NO_TARGET_FREE_INCREMENTAL_FAMILY"
        selected_family = None
        selected_raw = []
        selected_derived = []
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "TARGET_FREE_FAMILY_DECISION_V5",
        "status": status,
        "raw_component_order": list(RESPONSES),
        "raw_component_results": raw_results,
        "endpoint_veto_results": endpoint_results,
        "global_ambiguity": global_ambiguity,
        "global_ambiguity_reasons": reasons,
        "family_clear_pass_results": family_clear,
        "fixed_priority": list(FIXED_FAMILY_PRIORITY),
        "selected_family": selected_family,
        "selected_raw_columns": selected_raw,
        "selected_derived_columns": selected_derived,
        "all_threshold_float_hex_witnesses": thresholds,
        "all_tied_witnesses": ties,
        "no_label_network_future_year_model_submission_attestation": dict(
            FORBIDDEN_ACCESS_ATTESTATION
        ),
    }
    validate_family_decision_document(document)
    return document


def build_fit_ledger_document(
    raw_evaluations: Mapping[str, ResponseEvaluation],
    *,
    decision_status: str,
) -> dict[str, Any]:
    if tuple(raw_evaluations) != RESPONSES:
        raise IntegrityError("raw evaluation order or coverage differs")
    if decision_status not in FAMILY_DECISION_STATUSES:
        raise IntegrityError("fit ledger terminal status differs")
    completed: dict[str, Mapping[str, Any]] = {}
    skipped: dict[str, str] = {}
    for response in RESPONSES:
        evaluation = raw_evaluations[response]
        if evaluation.skip_reason is not None:
            skipped[response] = evaluation.skip_reason
        for item in evaluation.completed_fit_records:
            slot_id = str(item["planned_unit_slot_id"])
            if slot_id in completed:
                raise IntegrityError("duplicate completed fit slot")
            completed[slot_id] = item
    unit_slots = list(
        skipped_fit_ledger(
            skipped, completed_unit_records=completed
        )
    )
    primary = [
        item for item in unit_slots if item["unit_kind"] == "PRIMARY_MODEL"
    ]
    affine = [
        item for item in unit_slots if item["unit_kind"] == "ANALYTIC_AFFINE"
    ]
    completed_primary = sum(
        item["unit_status"] == UNIT_COMPLETED for item in primary
    )
    completed_affine = sum(
        item["unit_status"] == UNIT_COMPLETED for item in affine
    )
    skipped_primary = len(primary) - completed_primary
    skipped_affine = len(affine) - completed_affine
    method_counts = {
        "sklearn.pipeline.Pipeline.fit": sum(
            int(item["pipeline_fit_calls"]) for item in primary
        ),
        "sklearn.preprocessing.StandardScaler.fit": sum(
            int(item["standard_scaler_fit_calls"]) for item in primary
        ),
        "sklearn.linear_model.Ridge.fit": sum(
            int(item["ridge_fit_calls"]) for item in primary
        ),
        "sklearn.ensemble.ExtraTreesRegressor.fit": sum(
            int(item["extra_trees_fit_calls"]) for item in primary
        ),
    }
    method_counts["total_method_calls"] = sum(method_counts.values())
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "TARGET_FREE_FIT_LEDGER_V5",
        "status": decision_status,
        "planned_counts": {
            "primary_model_units": EXPECTED_PRIMARY_UNITS,
            "analytic_affine_units": EXPECTED_AFFINE_UNITS,
            "total_decision_units": EXPECTED_LEDGER_ROWS,
            "fit_ledger_rows": EXPECTED_LEDGER_ROWS,
        },
        "executed_counts": {
            "completed_primary_model_units": completed_primary,
            "completed_analytic_affine_units": completed_affine,
            "completed_total_decision_units": (
                completed_primary + completed_affine
            ),
        },
        "internal_method_call_counts": method_counts,
        "skipped_counts": {
            "skipped_primary_model_units": skipped_primary,
            "skipped_analytic_affine_units": skipped_affine,
            "skipped_total_decision_units": skipped_primary + skipped_affine,
        },
        "zero_fit_counts": {
            "derived": 0,
            "group": 0,
            "hyperparameter_or_seed_search": 0,
        },
        "unit_slots": unit_slots,
    }
    validate_fit_ledger_document(document)
    return document


def validate_fit_ledger_document(document: Mapping[str, Any]) -> None:
    record = _require_exact_keys(document, FIT_LEDGER_TOP_KEYS, "fit ledger")
    if (
        record["schema_version"] != SCHEMA_VERSION
        or record["artifact_type"] != "TARGET_FREE_FIT_LEDGER_V5"
        or record["status"] not in FAMILY_DECISION_STATUSES
    ):
        raise IntegrityError("fit ledger envelope differs")
    planned = _require_exact_keys(
        record["planned_counts"],
        FIT_LEDGER_PLANNED_COUNT_KEYS,
        "fit ledger planned counts",
    )
    if planned != {
        "primary_model_units": EXPECTED_PRIMARY_UNITS,
        "analytic_affine_units": EXPECTED_AFFINE_UNITS,
        "total_decision_units": EXPECTED_LEDGER_ROWS,
        "fit_ledger_rows": EXPECTED_LEDGER_ROWS,
    }:
        raise IntegrityError("fit ledger planned counts differ")
    executed = _require_exact_keys(
        record["executed_counts"],
        FIT_LEDGER_EXECUTED_COUNT_KEYS,
        "fit ledger executed counts",
    )
    methods = _require_exact_keys(
        record["internal_method_call_counts"],
        FIT_LEDGER_METHOD_COUNT_KEYS,
        "fit ledger method counts",
    )
    skipped = _require_exact_keys(
        record["skipped_counts"],
        FIT_LEDGER_SKIPPED_COUNT_KEYS,
        "fit ledger skipped counts",
    )
    zero = _require_exact_keys(
        record["zero_fit_counts"],
        FIT_LEDGER_ZERO_COUNT_KEYS,
        "fit ledger zero counts",
    )
    for mapping in (planned, executed, methods, skipped, zero):
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in mapping.values()
        ):
            raise IntegrityError("fit ledger count type or range differs")
    if zero != {
        "derived": 0,
        "group": 0,
        "hyperparameter_or_seed_search": 0,
    }:
        raise IntegrityError("fit ledger zero-fit counts differ")
    if methods["total_method_calls"] != sum(
        methods[key] for key in FIT_LEDGER_METHOD_COUNT_KEYS[:-1]
    ):
        raise IntegrityError("fit ledger method total differs")
    if (
        executed["completed_total_decision_units"]
        != executed["completed_primary_model_units"]
        + executed["completed_analytic_affine_units"]
        or skipped["skipped_total_decision_units"]
        != skipped["skipped_primary_model_units"]
        + skipped["skipped_analytic_affine_units"]
        or executed["completed_primary_model_units"]
        + skipped["skipped_primary_model_units"]
        != EXPECTED_PRIMARY_UNITS
        or executed["completed_analytic_affine_units"]
        + skipped["skipped_analytic_affine_units"]
        != EXPECTED_AFFINE_UNITS
        or executed["completed_total_decision_units"]
        + skipped["skipped_total_decision_units"]
        != EXPECTED_LEDGER_ROWS
    ):
        raise IntegrityError("fit ledger completed/skipped equations differ")
    slots = record["unit_slots"]
    if not isinstance(slots, list) or len(slots) != EXPECTED_LEDGER_ROWS:
        raise IntegrityError("fit ledger unit slot count differs")
    planned_units = planned_fit_units()
    observed_completed_primary = 0
    observed_completed_affine = 0
    observed_methods = {
        key: 0 for key in FIT_LEDGER_METHOD_COUNT_KEYS[:-1]
    }
    response_statuses: dict[str, set[str]] = {
        response: set() for response in RESPONSES
    }
    response_skip_reasons: dict[str, set[str]] = {
        response: set() for response in RESPONSES
    }
    affine_ordinal = 0
    for item, unit in zip(slots, planned_units, strict=True):
        if unit.unit_kind == "ANALYTIC_AFFINE":
            affine_ordinal += 1
        keys = (
            PRIMARY_LEDGER_KEYS
            if unit.unit_kind == "PRIMARY_MODEL"
            else AFFINE_LEDGER_KEYS
        )
        row = _require_exact_keys(item, keys, "fit ledger unit")
        if (
            row["planned_unit_slot_id"] != unit.planned_unit_slot_id
            or row["decision_fit_ordinal"] != unit.decision_fit_ordinal
            or row["unit_kind"] != unit.unit_kind
            or row["response"] != unit.response
            or row["fold"] != unit.fold
            or row["unit_status"] not in UNIT_STATUSES
            or row["fit_completed"]
            is not (row["unit_status"] == UNIT_COMPLETED)
        ):
            raise IntegrityError("fit ledger unit relation differs")
        response_statuses[unit.response].add(str(row["unit_status"]))
        if row["unit_status"] == UNIT_SKIPPED:
            reason = row["skip_reason"]
            if reason not in INVALID_OR_SKIP_REASON_SET:
                raise IntegrityError("fit ledger skip reason is outside exact13")
            response_skip_reasons[unit.response].add(str(reason))
            expected_skipped = (
                _skipped_primary_record(unit, str(reason))
                if unit.unit_kind == "PRIMARY_MODEL"
                else _skipped_affine_record(
                    unit, str(reason), affine_ordinal
                )
            )
            if dict(row) != expected_skipped:
                raise IntegrityError("fit ledger skipped unit differs")
            continue
        if row["skip_reason"] is not None:
            raise IntegrityError("completed fit ledger unit has skip reason")
        if unit.unit_kind == "PRIMARY_MODEL":
            observed_completed_primary += 1
            expected_input_dtype = (
                "float64_C_CONTIGUOUS"
                if unit.estimator_unit == "RIDGE_PIPELINE"
                else "float32_C_CONTIGUOUS"
            )
            expected_calls = (
                (1, 1, 1, 0, 1, None)
                if unit.estimator_unit == "RIDGE_PIPELINE"
                else (0, 0, 0, 1, 1, 260810)
            )
            observed_calls = (
                row["pipeline_fit_calls"],
                row["standard_scaler_fit_calls"],
                row["ridge_fit_calls"],
                row["extra_trees_fit_calls"],
                row["predict_calls"],
                row["random_state"],
            )
            if (
                row["estimator_unit"] != unit.estimator_unit
                or row["training_rows"] != 14_688
                or row["training_columns"] != len(PREDICTORS)
                or row["heldout_rows"] != 4_896
                or row["input_dtype"] != expected_input_dtype
                or row["response_dtype"] != "float64"
                or observed_calls != expected_calls
            ):
                raise IntegrityError("completed primary ledger unit differs")
            observed_methods["sklearn.pipeline.Pipeline.fit"] += int(
                row["pipeline_fit_calls"]
            )
            observed_methods[
                "sklearn.preprocessing.StandardScaler.fit"
            ] += int(row["standard_scaler_fit_calls"])
            observed_methods["sklearn.linear_model.Ridge.fit"] += int(
                row["ridge_fit_calls"]
            )
            observed_methods[
                "sklearn.ensemble.ExtraTreesRegressor.fit"
            ] += int(row["extra_trees_fit_calls"])
        else:
            observed_completed_affine += 1
            if (
                row["analytic_affine_ordinal"] != affine_ordinal
                or row["predictor"] != unit.predictor
                or row["training_rows"] != 14_688
                or row["heldout_rows"] != 4_896
                or not isinstance(row["constant_predictor_branch"], bool)
            ):
                raise IntegrityError("completed affine ledger unit differs")
            numeric_pairs = (
                ("sxx", "sxx_float_hex"),
                ("slope", "slope_float_hex"),
                ("intercept", "intercept_float_hex"),
            )
            for numeric_key, hex_key in numeric_pairs:
                if (
                    row[numeric_key] is None
                    or row[hex_key]
                    != canonical_float_hex(row[numeric_key])
                ):
                    raise IntegrityError(
                        "completed affine numeric witness differs"
                    )
            sxx = _finite_float64(row["sxx"], "affine ledger sxx")
            if bool(sxx < _np().float64(0.0)) or (
                row["constant_predictor_branch"]
                is not bool(sxx <= _np().float64(SXX_GUARD))
            ):
                raise IntegrityError("affine constant-predictor branch differs")
            if row["constant_predictor_branch"] and not _equal(
                row["slope"], 0.0
            ):
                raise IntegrityError("affine fallback slope is not zero")
    for response in RESPONSES:
        if len(response_statuses[response]) != 1:
            raise IntegrityError("fit ledger response is partially completed")
        if UNIT_SKIPPED in response_statuses[response] and len(
            response_skip_reasons[response]
        ) != 1:
            raise IntegrityError("fit ledger response skip reasons differ")
    observed_skipped_primary = EXPECTED_PRIMARY_UNITS - observed_completed_primary
    observed_skipped_affine = EXPECTED_AFFINE_UNITS - observed_completed_affine
    if executed != {
        "completed_primary_model_units": observed_completed_primary,
        "completed_analytic_affine_units": observed_completed_affine,
        "completed_total_decision_units": (
            observed_completed_primary + observed_completed_affine
        ),
    } or skipped != {
        "skipped_primary_model_units": observed_skipped_primary,
        "skipped_analytic_affine_units": observed_skipped_affine,
        "skipped_total_decision_units": (
            observed_skipped_primary + observed_skipped_affine
        ),
    }:
        raise IntegrityError("fit ledger aggregate counts differ from slots")
    observed_methods["total_method_calls"] = sum(observed_methods.values())
    if methods != observed_methods:
        raise IntegrityError("fit ledger method counts differ from slots")


def build_distribution_document(
    raw_evaluations: Mapping[str, ResponseEvaluation],
    derived: DerivedEvaluation,
    *,
    decision_status: str,
) -> dict[str, Any]:
    if tuple(raw_evaluations) != RESPONSES:
        raise IntegrityError("raw evaluation order or coverage differs")
    if decision_status not in FAMILY_DECISION_STATUSES:
        raise IntegrityError("distribution terminal status differs")
    records = [
        dict(item)
        for response in RESPONSES
        for item in raw_evaluations[response].distribution_records
    ]
    records.extend(dict(item) for item in derived.distribution_records)
    pooled = [dict(item) for item in derived.endpoint_pooled_records]
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "TARGET_FREE_DISTRIBUTION_STABILITY_V5",
        "status": decision_status,
        "scope_order": list(DISTRIBUTION_SCOPE_ORDER),
        "diagnostic_order": [*RESPONSES, *ENDPOINT_VETO_DIAGNOSTICS],
        "endpoint_pooled_metric_records": pooled,
        "records": records,
    }
    validate_distribution_document(document)
    return document


def validate_distribution_document(document: Mapping[str, Any]) -> None:
    record = _require_exact_keys(
        document, DISTRIBUTION_TOP_KEYS, "distribution evidence"
    )
    if (
        record["schema_version"] != SCHEMA_VERSION
        or record["artifact_type"]
        != "TARGET_FREE_DISTRIBUTION_STABILITY_V5"
        or record["status"] not in FAMILY_DECISION_STATUSES
        or record["scope_order"] != list(DISTRIBUTION_SCOPE_ORDER)
        or record["diagnostic_order"]
        != [*RESPONSES, *ENDPOINT_VETO_DIAGNOSTICS]
    ):
        raise IntegrityError("distribution evidence envelope differs")
    pooled = record["endpoint_pooled_metric_records"]
    if not isinstance(pooled, list) or len(pooled) != 3:
        raise IntegrityError("endpoint pooled record count differs")
    for item, diagnostic in zip(
        pooled, ENDPOINT_VETO_DIAGNOSTICS, strict=True
    ):
        row = _require_exact_keys(item, ENDPOINT_POOLED_KEYS, "endpoint pooled")
        if row["diagnostic"] != diagnostic:
            raise IntegrityError("endpoint pooled order differs")
        validate_validity_reason(
            row["valid"], row["invalid_reason"], "endpoint pooled"
        )
    rows = record["records"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_DISTRIBUTION_ROWS:
        raise IntegrityError("distribution record count differs")
    for item in rows:
        distribution_row = _require_exact_keys(
            item, DISTRIBUTION_RECORD_KEYS, "distribution row"
        )
        validate_validity_reason(
            distribution_row["valid"],
            distribution_row["invalid_reason"],
            "distribution row",
        )


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationBundle:
    raw_evaluations: Mapping[str, ResponseEvaluation]
    derived_evaluation: DerivedEvaluation
    decision_document: Mapping[str, Any]
    fit_ledger_document: Mapping[str, Any]
    distribution_document: Mapping[str, Any]
    duplicate_metric_records: tuple[dict[str, Any], ...]
    stability_records: tuple[dict[str, Any], ...]


def validate_evaluation_bundle_consistency(bundle: EvaluationBundle) -> None:
    """Crosslink every staged evidence tree before the first output write."""

    if tuple(bundle.raw_evaluations) != RESPONSES:
        raise IntegrityError("evaluation bundle raw response order differs")
    if (
        len(bundle.duplicate_metric_records) != EXPECTED_DUPLICATE_METRIC_ROWS
        or len(bundle.stability_records) != EXPECTED_STABILITY_ROWS
        or len(bundle.derived_evaluation.endpoint_results)
        != len(ENDPOINT_VETO_DIAGNOSTICS)
        or len(bundle.derived_evaluation.endpoint_pooled_records)
        != EXPECTED_ENDPOINT_POOLED_ROWS
        or len(bundle.derived_evaluation.stability_records)
        != len(ENDPOINT_VETO_DIAGNOSTICS) * 29
        or len(bundle.derived_evaluation.distribution_records)
        != len(ENDPOINT_VETO_DIAGNOSTICS) * 52
    ):
        raise IntegrityError("evaluation bundle fixed record counts differ")
    decision = bundle.decision_document
    validate_family_decision_document(decision)
    decision_raw = decision["raw_component_results"]
    for response, decision_row in zip(RESPONSES, decision_raw, strict=True):
        evaluation = bundle.raw_evaluations[response]
        if evaluation.response != response or dict(evaluation.raw_result) != dict(
            decision_row
        ):
            raise IntegrityError("raw evaluation and decision crosslink differs")
        premodel = decision_row["classification"] in PREMODEL_CLASSIFICATIONS
        if (evaluation.skip_reason is not None) is not premodel:
            raise IntegrityError("raw premodel skip relation differs")
        if premodel:
            if (
                evaluation.skip_reason not in INVALID_OR_SKIP_REASON_SET
                or evaluation.completed_fit_records
                or evaluation.ridge_oof is not None
                or evaluation.extra_trees_oof is not None
                or evaluation.independent_oof is not None
            ):
                raise IntegrityError("premodel response retained fit evidence")
        elif (
            len(evaluation.completed_fit_records) != 148
            or evaluation.ridge_oof is None
            or evaluation.extra_trees_oof is None
        ):
            raise IntegrityError("evaluable response fit evidence differs")
        if len(evaluation.duplicate_metric_records) != 37:
            raise IntegrityError("per-response duplicate metric slots differ")
        expected_boundary_text = ";".join(
            decision_row["boundary_equality_flags"]
        )
        validate_csv_boundary_flags_text(expected_boundary_text)
        if any(
            item["boundary_equality_flags"] != expected_boundary_text
            or item["component_classification"]
            != decision_row["classification"]
            for item in evaluation.duplicate_metric_records
        ):
            raise IntegrityError(
                "duplicate metric boundary/classification projection differs"
            )
        if len(evaluation.stability_records) != 29:
            raise IntegrityError("per-response stability slots differ")
        if len(evaluation.distribution_records) != 52:
            raise IntegrityError("per-response distribution slots differ")
    if tuple(
        dict(item) for item in bundle.derived_evaluation.endpoint_results
    ) != tuple(dict(item) for item in decision["endpoint_veto_results"]):
        raise IntegrityError("endpoint evaluation and decision crosslink differs")
    expected_metrics = tuple(
        item
        for response in RESPONSES
        for item in bundle.raw_evaluations[response].duplicate_metric_records
    )
    expected_stability = tuple(
        [
            item
            for response in RESPONSES
            for item in bundle.raw_evaluations[response].stability_records
        ]
        + list(bundle.derived_evaluation.stability_records)
    )
    if bundle.duplicate_metric_records != expected_metrics:
        raise IntegrityError("duplicate metric bundle projection differs")
    if bundle.stability_records != expected_stability:
        raise IntegrityError("stability bundle projection differs")
    distribution = bundle.distribution_document
    validate_distribution_document(distribution)
    expected_distribution = [
        dict(item)
        for response in RESPONSES
        for item in bundle.raw_evaluations[response].distribution_records
    ] + [dict(item) for item in bundle.derived_evaluation.distribution_records]
    if distribution["records"] != expected_distribution or distribution[
        "endpoint_pooled_metric_records"
    ] != [
        dict(item)
        for item in bundle.derived_evaluation.endpoint_pooled_records
    ]:
        raise IntegrityError("distribution bundle projection differs")
    ledger = bundle.fit_ledger_document
    validate_fit_ledger_document(ledger)
    if (
        ledger["status"] != decision["status"]
        or distribution["status"] != decision["status"]
    ):
        raise IntegrityError("bundle terminal status crosslink differs")
    slots_by_response: dict[str, list[Mapping[str, Any]]] = {
        response: [] for response in RESPONSES
    }
    for slot in ledger["unit_slots"]:
        slots_by_response[str(slot["response"])].append(slot)
    for response in RESPONSES:
        evaluation = bundle.raw_evaluations[response]
        expected_status = (
            UNIT_SKIPPED
            if evaluation.skip_reason is not None
            else UNIT_COMPLETED
        )
        slots = slots_by_response[response]
        if len(slots) != 148 or any(
            slot["unit_status"] != expected_status for slot in slots
        ):
            raise IntegrityError("fit slots and response evaluation differ")
        if expected_status == UNIT_SKIPPED and any(
            slot["skip_reason"] != evaluation.skip_reason for slot in slots
        ):
            raise IntegrityError("fit slot skip reason crosslink differs")
        selector = evaluation.independent_estimator
        selected_primary = [
            slot
            for slot in slots
            if slot["unit_kind"] == "PRIMARY_MODEL"
            and slot["estimator_unit"] == selector
        ] if selector is not None else []
        if selector is None:
            if evaluation.independent_oof is not None:
                raise IntegrityError(
                    "null independent selector retained selected OOF values"
                )
        elif (
            selector not in {"RIDGE_PIPELINE", "EXTRA_TREES"}
            or len(selected_primary) != len(FOLDS)
            or [slot["fold"] for slot in selected_primary]
            != [fold.name for fold in FOLDS]
            or any(
                slot["unit_status"] != UNIT_COMPLETED
                for slot in selected_primary
            )
            or evaluation.independent_oof is None
        ):
            raise IntegrityError(
                "independent selector lacks its exact four completed primary slots"
            )


def evaluate_prepared_execution(prepared: PreparedExecution) -> EvaluationBundle:
    prepared.authorization.assert_arrow_handoff()
    raw = {
        response: evaluate_raw_response(prepared, response)
        for response in RESPONSES
    }
    if tuple(raw) != RESPONSES:
        raise AssertionError("raw evaluation order drift")
    derived = evaluate_derived_wind(prepared, raw)
    decision = build_family_decision_document(raw, derived)
    ledger = build_fit_ledger_document(
        raw, decision_status=str(decision["status"])
    )
    distribution = build_distribution_document(
        raw, derived, decision_status=str(decision["status"])
    )
    metrics = tuple(
        item
        for response in RESPONSES
        for item in raw[response].duplicate_metric_records
    )
    stability = tuple(
        [
            item
            for response in RESPONSES
            for item in raw[response].stability_records
        ]
        + list(derived.stability_records)
    )
    if len(metrics) != EXPECTED_DUPLICATE_METRIC_ROWS:
        raise AssertionError("duplicate metric row count drift")
    if len(stability) != EXPECTED_STABILITY_ROWS:
        raise AssertionError("stability row count drift")
    bundle = EvaluationBundle(
        raw_evaluations=raw,
        derived_evaluation=derived,
        decision_document=decision,
        fit_ledger_document=ledger,
        distribution_document=distribution,
        duplicate_metric_records=metrics,
        stability_records=stability,
    )
    validate_evaluation_bundle_consistency(bundle)
    return bundle


CSV_BOOLEAN_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "valid",
        "independent_selected",
        "parent_max_r2_witness",
        "parent_redundancy_boolean",
        "affine_qualifies",
        "r2_boundary_equal_0_995",
        "nrmse_boundary_equal_0_02",
        "weak_floor_applicable",
        "nrmse_boundary_equal_0_01",
        "strict_not_exact_pass",
        "weak_floor_pass",
    }
)


def _csv_cell(value: Any) -> str:
    np = _np()
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if bool(value) else "FALSE"
    if isinstance(value, (float, np.floating)):
        converted = _finite_float64(value, "CSV float")
        if bool(converted == np.float64(0.0)):
            converted = np.float64(0.0)
        return format(float(converted), ".17g")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (dict, list, tuple)):
        return strict_json_dumps(value, pretty=False).decode("ascii")
    text = str(value)
    if text.lower() in {"nan", "inf", "-inf", "infinity", "-infinity"}:
        raise StrictJSONError("forbidden non-finite CSV literal")
    return text


def validate_csv_boundary_flags_text(value: Any) -> str:
    if not isinstance(value, str):
        raise IntegrityError("CSV boundary equality flags must be a string")
    if value == "":
        return value
    names = value.split(";")
    if (
        any(not name for name in names)
        or len(names) != len(set(names))
        or any(name not in CUTOFF_NAMES for name in names)
        or names != [name for name in CUTOFF_NAMES if name in names]
    ):
        raise IntegrityError(
            "CSV boundary equality flags are unknown, inactive, duplicate, or out of order"
        )
    return value


def serialize_csv_rows(
    rows: Iterable[Mapping[str, Any]], columns: Sequence[str]
) -> bytes:
    frozen_columns = tuple(columns)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(frozen_columns)
    count = 0
    for row in rows:
        if set(row) != set(frozen_columns) or len(row) != len(frozen_columns):
            raise IntegrityError("CSV row schema differs from frozen columns")
        cells = []
        for column in frozen_columns:
            value = row[column]
            if column in CSV_BOOLEAN_COLUMNS and value is not None and not isinstance(
                value, (bool, _np().bool_)
            ):
                raise IntegrityError("CSV boolean cell is not a boolean or null")
            if column == "boundary_equality_flags":
                value = validate_csv_boundary_flags_text(value)
            cells.append(_csv_cell(value))
        writer.writerow(cells)
        count += 1
    payload = buffer.getvalue().encode("utf-8")
    if payload.startswith(b"\xef\xbb\xbf") or b"\r\n" in payload:
        raise IntegrityError("CSV encoding or newline contract drift")
    if count < 0:
        raise AssertionError("unreachable CSV row count")
    return payload


@dataclasses.dataclass(frozen=True, slots=True)
class StagedIdentity:
    path: str
    size_bytes: int
    sha256: str
    format: str
    row_count: int
    logical_sha256: str


def _assert_output_path_no_linklike(path: Path, label: str) -> None:
    try:
        _assert_no_linklike_absolute(path, label)
    except IdentityError as exc:
        raise OutputPublicationError(str(exc)) from exc


def _exclusive_write_bytes(path: Path, payload: bytes) -> tuple[int, str]:
    if _lexists(path):
        raise OutputPublicationError(f"create-if-absent path exists: {path}")
    _assert_output_path_no_linklike(path.parent, "exclusive-write parent")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise OutputPublicationError(f"exclusive write failed: {path}") from exc
    size, digest = _sha256_and_size(path)
    if size != len(payload) or digest != hashlib.sha256(payload).hexdigest():
        raise OutputPublicationError(f"post-write identity differs: {path}")
    return size, digest


def arrow_logical_sha256(table: Any, expected_schema: Any) -> str:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    if not table.schema.equals(expected_schema, check_metadata=True):
        raise IntegrityError("Arrow table schema differs before logical hash")
    combined = table.combine_chunks()
    sink = pa.BufferOutputStream()
    options = ipc.IpcWriteOptions(metadata_version=ipc.MetadataVersion.V5)
    with ipc.new_stream(sink, expected_schema, options=options) as writer:
        writer.write_table(combined)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _validate_v4_parquet_schema(schema: Any) -> None:
    import pyarrow as pa

    names = tuple(schema.names)
    for name, expected_type in (
        ("valid_time_utc", pa.timestamp("ns", tz="UTC")),
        ("target_operating_day_kst", pa.date32()),
    ):
        if names.count(name) != 1:
            raise IntegrityError(f"V4 Parquet timestamp/date field differs: {name}")
        field = schema.field(name)
        if field.nullable or field.type != expected_type:
            raise IntegrityError(f"V4 Parquet timestamp/date type differs: {name}")
    for field in schema:
        if pa.types.is_floating(field.type) and field.type != pa.float64():
            raise IntegrityError("V4 Parquet floating column is not float64")
    if b"pandas" in (schema.metadata or {}):
        raise IntegrityError("pandas schema metadata is forbidden")


def write_parquet_exclusive(
    path: Path,
    *,
    schema: Any,
    arrays: Sequence[Any],
    row_count: int,
) -> StagedIdentity:
    """Write an explicitly schematized Parquet file without pandas metadata."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    if _lexists(path):
        raise OutputPublicationError(f"create-if-absent path exists: {path}")
    _assert_output_path_no_linklike(path.parent, "Parquet-write parent")
    if len(arrays) != len(schema):
        raise IntegrityError("Parquet array count differs from schema")
    _validate_v4_parquet_schema(schema)
    table = pa.Table.from_arrays(list(arrays), schema=schema)
    if table.num_rows != row_count:
        raise IntegrityError("Parquet table row count differs")
    try:
        with path.open("xb") as handle:
            pq.write_table(
                table,
                handle,
                **dict(PARQUET_WRITE_TABLE_KWARGS_EXACT),
            )
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise OutputPublicationError(f"Parquet exclusive write failed: {path}") from exc
    parquet_file = pq.ParquetFile(path)
    try:
        footer_schema = parquet_file.schema_arrow
        metadata = parquet_file.metadata
        roundtrip = parquet_file.read(use_threads=False)
    finally:
        parquet_file.close()
    if (
        not footer_schema.equals(schema, check_metadata=True)
        or metadata.num_rows != row_count
        or metadata.num_row_groups != (row_count + 65_535) // 65_536
        or not roundtrip.schema.equals(schema, check_metadata=True)
        or not roundtrip.equals(table, check_metadata=True)
    ):
        raise IntegrityError("V4 Parquet footer or full-table roundtrip differs")
    roundtrip_logical_sha = arrow_logical_sha256(roundtrip, schema)
    del roundtrip, table
    size, digest = _sha256_and_size(path)
    return StagedIdentity(
        path=path.as_posix(),
        size_bytes=size,
        sha256=digest,
        format="PARQUET",
        row_count=row_count,
        logical_sha256=roundtrip_logical_sha,
    )


def stage_serialized_runner_artifacts(
    artifact_root: Path,
    payloads: Mapping[str, tuple[bytes, str, int, str]],
    *,
    authorization: PredataAuthorization,
) -> tuple[StagedIdentity, ...]:
    """Create the exact seven runner artifacts in a fresh transaction dir."""

    authorization.assert_ready()
    if tuple(payloads) != RUNNER_ARTIFACT_BASENAMES:
        raise OutputPublicationError("runner artifact order or exact set differs")
    root = artifact_root.resolve(strict=True)
    output_root = root / OUTPUT_ROOT_RELATIVE
    if _linklike(output_root) or not output_root.is_dir():
        raise OutputPublicationError("V5 output root is absent")
    transaction = output_root / TRANSACTION_DIRNAME
    if _lexists(transaction):
        raise OutputPublicationError("runner transaction root already exists")
    _assert_output_path_no_linklike(output_root, "runner output root")
    try:
        transaction.mkdir(exist_ok=False)
    except OSError as exc:
        raise OutputPublicationError("runner transaction creation failed") from exc
    identities: list[StagedIdentity] = []
    for basename in RUNNER_ARTIFACT_BASENAMES:
        payload, format_name, row_count, logical_sha = payloads[basename]
        path = transaction / basename
        size, digest = _exclusive_write_bytes(path, payload)
        if format_name in {"CSV", "JSON"} and logical_sha != digest:
            raise OutputPublicationError(
                "CSV and JSON logical identity must equal physical identity"
            )
        identities.append(
            StagedIdentity(
                path=path.relative_to(root).as_posix(),
                size_bytes=size,
                sha256=digest,
                format=format_name,
                row_count=row_count,
                logical_sha256=logical_sha,
            )
        )
    if len(identities) != 7:
        raise AssertionError("runner staged artifact count drift")
    return tuple(identities)


def _relative_staged_identity(
    identity: StagedIdentity, root: Path, path: Path
) -> StagedIdentity:
    return dataclasses.replace(
        identity, path=path.relative_to(root).as_posix()
    )


def _stage_bytes_identity(
    root: Path,
    path: Path,
    payload: bytes,
    *,
    format_name: str,
    row_count: int,
) -> StagedIdentity:
    size, digest = _exclusive_write_bytes(path, payload)
    return StagedIdentity(
        path=path.relative_to(root).as_posix(),
        size_bytes=size,
        sha256=digest,
        format=format_name,
        row_count=row_count,
        logical_sha256=digest,
    )


def stage_evaluation_bundle(
    prepared: PreparedExecution, bundle: EvaluationBundle
) -> tuple[StagedIdentity, ...]:
    """Stage the exact seven runner artifacts, leaving publication to sealer."""

    prepared.authorization.assert_arrow_handoff()
    root = prepared.artifact_root.resolve(strict=True)
    output_root = root / OUTPUT_ROOT_RELATIVE
    expected_aliases = {
        Path(spec.final_relative_path).name for spec in ALIAS_SPECS
    }
    transaction = output_root / TRANSACTION_DIRNAME
    if _lexists(transaction):
        raise OutputPublicationError("runner transaction root already exists")
    if _linklike(output_root) or not output_root.is_dir():
        raise OutputPublicationError(
            "pre-staging output root must contain exactly the two aliases"
        )
    output_entries = tuple(output_root.iterdir())
    if any(_linklike(item) for item in output_entries) or {
        item.name for item in output_entries
    } != expected_aliases:
        raise OutputPublicationError(
            "pre-staging output root must contain exactly the two aliases"
        )
    if {
        Path(identity.path).name for identity in prepared.alias_identities
    } != expected_aliases:
        raise OutputPublicationError("prepared alias identity set differs")
    _assert_output_path_no_linklike(output_root, "runner output root")
    try:
        transaction.mkdir(exist_ok=False)
    except OSError as exc:
        raise OutputPublicationError(
            "runner transaction create-if-absent failed"
        ) from exc

    decision = bundle.decision_document
    status = str(decision["status"])
    validate_evaluation_bundle_consistency(bundle)
    if bundle.fit_ledger_document["status"] != status or bundle.distribution_document[
        "status"
    ] != status:
        raise IntegrityError("staged evidence terminal status crosslink differs")
    validate_family_decision_document(decision)
    validate_fit_ledger_document(bundle.fit_ledger_document)
    validate_distribution_document(bundle.distribution_document)
    for row in bundle.duplicate_metric_records:
        _require_exact_keys(row, DUPLICATE_METRIC_COLUMNS, "duplicate metric row")
        validate_validity_reason(
            row["valid"], row["invalid_reason"], "duplicate metric row"
        )
    for row in bundle.stability_records:
        _require_exact_keys(row, STABILITY_COLUMNS, "stability row")
        validate_validity_reason(
            row["valid"], row["invalid_reason"], "stability row"
        )
    metrics_payload = serialize_csv_rows(
        bundle.duplicate_metric_records, DUPLICATE_METRIC_COLUMNS
    )
    stability_payload = serialize_csv_rows(
        bundle.stability_records, STABILITY_COLUMNS
    )
    decision_payload = strict_json_dumps(decision, pretty=True)
    fit_payload = strict_json_dumps(bundle.fit_ledger_document, pretty=True)
    distribution_payload = strict_json_dumps(
        bundle.distribution_document, pretty=True
    )

    identities: list[StagedIdentity] = []
    identities.append(
        _stage_bytes_identity(
            root,
            transaction / RUNNER_ARTIFACT_BASENAMES[0],
            metrics_payload,
            format_name="CSV",
            row_count=EXPECTED_DUPLICATE_METRIC_ROWS,
        )
    )
    identities.append(
        _stage_bytes_identity(
            root,
            transaction / RUNNER_ARTIFACT_BASENAMES[1],
            stability_payload,
            format_name="CSV",
            row_count=EXPECTED_STABILITY_ROWS,
        )
    )
    identities.append(
        _stage_bytes_identity(
            root,
            transaction / RUNNER_ARTIFACT_BASENAMES[2],
            decision_payload,
            format_name="JSON",
            row_count=1,
        )
    )
    oof_schema, oof_arrays = build_oof_arrow_table(
        prepared, bundle.raw_evaluations
    )
    oof_path = transaction / RUNNER_ARTIFACT_BASENAMES[3]
    identities.append(
        _relative_staged_identity(
            write_parquet_exclusive(
                oof_path,
                schema=oof_schema,
                arrays=oof_arrays,
                row_count=EXPECTED_OOF_ROWS,
            ),
            root,
            oof_path,
        )
    )
    del oof_arrays
    identities.append(
        _stage_bytes_identity(
            root,
            transaction / RUNNER_ARTIFACT_BASENAMES[4],
            fit_payload,
            format_name="JSON",
            row_count=EXPECTED_LEDGER_ROWS,
        )
    )
    identities.append(
        _stage_bytes_identity(
            root,
            transaction / RUNNER_ARTIFACT_BASENAMES[5],
            distribution_payload,
            format_name="JSON",
            row_count=EXPECTED_DISTRIBUTION_ROWS,
        )
    )
    derived_schema, derived_arrays = build_derived_arrow_table(
        prepared, bundle.raw_evaluations, bundle.derived_evaluation
    )
    derived_path = transaction / RUNNER_ARTIFACT_BASENAMES[6]
    identities.append(
        _relative_staged_identity(
            write_parquet_exclusive(
                derived_path,
                schema=derived_schema,
                arrays=derived_arrays,
                row_count=EXPECTED_DERIVED_ROWS,
            ),
            root,
            derived_path,
        )
    )
    del derived_arrays
    if [Path(item.path).name for item in identities] != list(
        RUNNER_ARTIFACT_BASENAMES
    ):
        raise AssertionError("staged runner identity order drift")
    transaction_entries = tuple(transaction.iterdir())
    if any(_linklike(item) for item in transaction_entries) or {
        item.name for item in transaction_entries
    } != set(RUNNER_ARTIFACT_BASENAMES):
        raise OutputPublicationError("runner transaction exact set differs")
    root_entries = {item.name for item in output_root.iterdir()}
    if root_entries != {*expected_aliases, TRANSACTION_DIRNAME}:
        raise OutputPublicationError("post-staging output root exact set differs")
    for basename in (*SEALER_ARTIFACT_BASENAMES, *RUNNER_ARTIFACT_BASENAMES):
        if _lexists(output_root / basename):
            raise OutputPublicationError(
                "runner must not create a final publication artifact"
            )
    return tuple(identities)


RUNNER_STDOUT_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "artifact_type",
    "status",
    "attempt_id",
    "decision",
    "output_counts",
    "staged_output_identities",
    "threadpool_info_before_inside_after",
    "no_forbidden_access_attestation",
    "manifest_present",
    "family_lock_present",
)
RUNNER_OUTPUT_COUNT_KEYS: Final[tuple[str, ...]] = (
    "aliases",
    "runner_staged_outputs",
    "metrics_rows",
    "stability_rows",
    "oof_rows",
    "fit_ledger_rows",
    "distribution_records",
    "endpoint_pooled_metric_records",
    "derived_rows",
)


@dataclasses.dataclass(frozen=True, slots=True)
class RunnerResult:
    attempt_id: str
    decision_document: Mapping[str, Any]
    staged_output_identities: tuple[StagedIdentity, ...]
    threadpool_evidence: Mapping[str, Any]

    def stdout_document(self) -> dict[str, Any]:
        decision = self.decision_document
        document = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "TARGET_FREE_DUPLICATE_RUNNER_STAGED_RESULT_V5",
            "status": "RUNNER_STAGED_OUTPUTS_COMPLETE_PENDING_POSTRUN_AUDIT",
            "attempt_id": self.attempt_id,
            "decision": {
                "status": decision["status"],
                "selected_family": decision["selected_family"],
                "global_ambiguity": decision["global_ambiguity"],
            },
            "output_counts": {
                "aliases": 2,
                "runner_staged_outputs": 7,
                "metrics_rows": EXPECTED_DUPLICATE_METRIC_ROWS,
                "stability_rows": EXPECTED_STABILITY_ROWS,
                "oof_rows": EXPECTED_OOF_ROWS,
                "fit_ledger_rows": EXPECTED_LEDGER_ROWS,
                "distribution_records": EXPECTED_DISTRIBUTION_ROWS,
                "endpoint_pooled_metric_records": EXPECTED_ENDPOINT_POOLED_ROWS,
                "derived_rows": EXPECTED_DERIVED_ROWS,
            },
            "staged_output_identities": [
                dataclasses.asdict(identity)
                for identity in self.staged_output_identities
            ],
            "threadpool_info_before_inside_after": dict(
                self.threadpool_evidence
            ),
            "no_forbidden_access_attestation": dict(
                FORBIDDEN_ACCESS_ATTESTATION
            ),
            "manifest_present": False,
            "family_lock_present": False,
        }
        if tuple(document) != RUNNER_STDOUT_KEYS:
            raise AssertionError("runner stdout schema drift")
        if tuple(document["output_counts"]) != RUNNER_OUTPUT_COUNT_KEYS:
            raise AssertionError("runner output count schema drift")
        validate_runner_stdout_document(document)
        return document


def validate_runner_stdout_document(document: Mapping[str, Any]) -> None:
    record = _require_exact_keys(
        document, RUNNER_STDOUT_KEYS, "runner stdout"
    )
    if (
        record["schema_version"] != SCHEMA_VERSION
        or record["artifact_type"]
        != "TARGET_FREE_DUPLICATE_RUNNER_STAGED_RESULT_V5"
        or record["status"]
        != "RUNNER_STAGED_OUTPUTS_COMPLETE_PENDING_POSTRUN_AUDIT"
    ):
        raise IntegrityError("runner stdout envelope differs")
    validate_attempt_id(record["attempt_id"])
    decision = _require_exact_keys(
        record["decision"],
        ("status", "selected_family", "global_ambiguity"),
        "runner stdout decision",
    )
    if (
        decision["status"] not in FAMILY_DECISION_STATUSES
        or decision["selected_family"]
        not in {None, *FIXED_FAMILY_PRIORITY}
        or not isinstance(decision["global_ambiguity"], bool)
    ):
        raise IntegrityError("runner stdout decision differs")
    if decision["global_ambiguity"]:
        if (
            decision["status"] != "STOP_FAMILY_AMBIGUOUS"
            or decision["selected_family"] is not None
        ):
            raise IntegrityError("runner ambiguity decision differs")
    elif decision["status"] == "STOP_FAMILY_AMBIGUOUS":
        raise IntegrityError("runner ambiguity status lacks ambiguity flag")
    counts = _require_exact_keys(
        record["output_counts"], RUNNER_OUTPUT_COUNT_KEYS, "runner counts"
    )
    if counts != {
        "aliases": 2,
        "runner_staged_outputs": 7,
        "metrics_rows": EXPECTED_DUPLICATE_METRIC_ROWS,
        "stability_rows": EXPECTED_STABILITY_ROWS,
        "oof_rows": EXPECTED_OOF_ROWS,
        "fit_ledger_rows": EXPECTED_LEDGER_ROWS,
        "distribution_records": EXPECTED_DISTRIBUTION_ROWS,
        "endpoint_pooled_metric_records": EXPECTED_ENDPOINT_POOLED_ROWS,
        "derived_rows": EXPECTED_DERIVED_ROWS,
    }:
        raise IntegrityError("runner output counts differ")
    identities = record["staged_output_identities"]
    if not isinstance(identities, list) or len(identities) != 7:
        raise IntegrityError("runner staged identity count differs")
    row_counts = (
        EXPECTED_DUPLICATE_METRIC_ROWS,
        EXPECTED_STABILITY_ROWS,
        1,
        EXPECTED_OOF_ROWS,
        EXPECTED_LEDGER_ROWS,
        EXPECTED_DISTRIBUTION_ROWS,
        EXPECTED_DERIVED_ROWS,
    )
    for item, basename, row_count in zip(
        identities, RUNNER_ARTIFACT_BASENAMES, row_counts, strict=True
    ):
        identity = _require_exact_keys(
            item,
            ("path", "size_bytes", "sha256", "format", "row_count", "logical_sha256"),
            "runner staged identity",
        )
        expected_path = (
            OUTPUT_ROOT_RELATIVE
            + "/"
            + TRANSACTION_DIRNAME
            + "/"
            + basename
        )
        expected_format = (
            "PARQUET"
            if basename.endswith(".parquet")
            else "CSV" if basename.endswith(".csv") else "JSON"
        )
        if (
            identity["path"] != expected_path
            or not isinstance(identity["size_bytes"], int)
            or isinstance(identity["size_bytes"], bool)
            or identity["size_bytes"] <= 0
            or re.fullmatch(r"[0-9a-f]{64}", str(identity["sha256"]))
            is None
            or identity["format"] != expected_format
            or identity["row_count"] != row_count
            or re.fullmatch(
                r"[0-9a-f]{64}", str(identity["logical_sha256"])
            )
            is None
            or (
                expected_format in {"CSV", "JSON"}
                and identity["logical_sha256"] != identity["sha256"]
            )
        ):
            raise IntegrityError("runner staged identity differs")
    threadpool = _require_exact_keys(
        record["threadpool_info_before_inside_after"],
        THREADPOOL_EVIDENCE_KEYS,
        "runner threadpool evidence",
    )
    if threadpool["capture_phases"] != list(THREADPOOL_PHASES):
        raise IntegrityError("runner threadpool phases differ")
    phase_counts = _require_exact_keys(
        threadpool["capture_counts"], THREADPOOL_PHASES, "runner thread counts"
    )
    if (
        phase_counts["BEFORE_EXPLICIT_CONTEXT"] != 1
        or phase_counts["INSIDE_CONTEXT"] != phase_counts["AFTER_CONTEXT"]
        or threadpool["event_count"]
        != 1 + 2 * phase_counts["INSIDE_CONTEXT"]
        or re.fullmatch(
            r"[0-9a-f]{64}", str(threadpool["event_chain_sha256"])
        )
        is None
    ):
        raise IntegrityError("runner threadpool aggregate differs")
    for key in (
        "first_observation_by_phase",
        "last_observation_by_phase",
    ):
        boundaries = _require_exact_keys(
            threadpool[key], THREADPOOL_PHASES, "runner thread boundaries"
        )
        for phase in THREADPOOL_PHASES:
            event = boundaries[phase]
            if phase != "BEFORE_EXPLICIT_CONTEXT" and phase_counts[
                "INSIDE_CONTEXT"
            ] == 0:
                if event is not None:
                    raise IntegrityError("zero-fit runner thread boundary differs")
                continue
            item = _require_exact_keys(
                event, THREADPOOL_EVENT_KEYS, "runner thread event"
            )
            if item["phase"] != phase:
                raise IntegrityError("runner thread boundary phase differs")
            _validate_persisted_threadpool_observation(item["threadpools"])
    if record["no_forbidden_access_attestation"] != dict(
        FORBIDDEN_ACCESS_ATTESTATION
    ) or record["manifest_present"] is not False or record[
        "family_lock_present"
    ] is not False:
        raise IntegrityError("runner forbidden/final-publication attestation differs")


def run_prepared_execution(prepared: PreparedExecution) -> RunnerResult:
    bundle = evaluate_prepared_execution(prepared)
    if prepared.authorization.threadpool_recorder is None:
        raise PredataAuthorizationError("threadpool recorder is unavailable")
    threadpool_evidence = (
        prepared.authorization.threadpool_recorder.snapshot()
    )
    validate_threadpool_evidence(
        threadpool_evidence, bundle.fit_ledger_document
    )
    staged = stage_evaluation_bundle(prepared, bundle)
    return RunnerResult(
        attempt_id=prepared.attempt_id,
        decision_document=bundle.decision_document,
        staged_output_identities=staged,
        threadpool_evidence=threadpool_evidence,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class AliasSpec:
    source_identity: FileIdentity
    stage_relative_path: str
    final_relative_path: str


ALIAS_SPECS: Final[tuple[AliasSpec, ...]] = (
    AliasSpec(
        source_identity=FileIdentity(
            "manifest_census_v1.json",
            4_523,
            "0269f6717b1110fafb6bba052c78824c616c648dbbdfe9f630947012c3f571a1",
        ),
        stage_relative_path=(
            "modeling/target_free_multiseason_v5/"
            ".alias_stage_FIELD_CENSUS_LOCK_v5"
        ),
        final_relative_path=(
            "modeling/target_free_multiseason_v5/FIELD_CENSUS_LOCK.json"
        ),
    ),
    AliasSpec(
        source_identity=FileIdentity(
            "raw/RAW_RANGE_MANIFEST.parquet",
            3_812_043,
            "11849682fd1c65ca20cfcb3002d3708beccf10ef862bb0f03d054c2c030790d5",
        ),
        stage_relative_path=(
            "modeling/target_free_multiseason_v5/"
            ".alias_stage_PROVENANCE_LEDGER_v5"
        ),
        final_relative_path=(
            "modeling/target_free_multiseason_v5/PROVENANCE_LEDGER.parquet"
        ),
    ),
)


def assert_v5_namespace_absent(artifact_root: Path) -> Path:
    root = artifact_root.resolve(strict=True)
    validate_runner_zero_state(root)
    output_root = root / OUTPUT_ROOT_RELATIVE
    _assert_output_path_no_linklike(
        output_root.parent, "V5 output namespace parent"
    )
    if _lexists(output_root):
        raise OutputPublicationError("V5 output namespace must be absent")
    return output_root


def create_v5_namespace_exclusive(artifact_root: Path) -> Path:
    output_root = assert_v5_namespace_absent(artifact_root)
    try:
        output_root.mkdir(parents=False, exist_ok=False)
    except OSError as exc:
        raise OutputPublicationError("exclusive V5 namespace creation failed") from exc
    if _linklike(output_root):
        raise OutputPublicationError("created V5 namespace is link-like")
    return output_root


def publish_exact_byte_alias(
    artifact_root: Path,
    spec: AliasSpec,
    *,
    authorization: PredataAuthorization,
    copy_chunk_size: int = 1 << 20,
) -> FileIdentity:
    """Stage-copy, rehash, hardlink final, unlink stage, then prove separation."""

    authorization.assert_alias_publication_ready()
    root = artifact_root.resolve(strict=True)
    source = revalidate_file_identity(root, spec.source_identity)
    stage = root / spec.stage_relative_path
    final = root / spec.final_relative_path
    if _lexists(stage) or _lexists(final):
        raise OutputPublicationError("alias stage or final already exists")
    if (
        stage.parent != final.parent
        or _linklike(stage.parent)
        or not stage.parent.is_dir()
    ):
        raise OutputPublicationError("alias parent namespace is unavailable")
    _assert_output_path_no_linklike(stage.parent, "alias parent namespace")
    try:
        with source.open("rb") as reader, stage.open("xb") as writer:
            while True:
                chunk = reader.read(copy_chunk_size)
                if not chunk:
                    break
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        stage_size, stage_sha = _sha256_and_size(stage)
        if (
            stage_size != spec.source_identity.size_bytes
            or stage_sha != spec.source_identity.sha256
        ):
            raise OutputPublicationError("staged alias byte identity differs")
        os.link(stage, final)
        stage.unlink()
        final_size, final_sha = _sha256_and_size(final)
        if final_size != stage_size or final_sha != stage_sha:
            raise OutputPublicationError("final alias byte identity differs")
        if os.path.samefile(source, final):
            raise OutputPublicationError("persistent alias is linked to source inode")
        source_stat = source.stat()
        final_stat = final.stat()
        if source_stat.st_ino and final_stat.st_ino and source_stat.st_ino == final_stat.st_ino:
            raise OutputPublicationError("source and final file IDs are equal")
        if source_stat.st_nlink != 1 or final_stat.st_nlink != 1:
            raise OutputPublicationError("persistent link count differs from one")
    except OutputPublicationError:
        raise
    except OSError as exc:
        raise OutputPublicationError(
            "alias publication failed; a V5 incident and V6 are required"
        ) from exc
    return FileIdentity(spec.final_relative_path, final_size, final_sha)


def validate_published_alias_closure(
    artifact_root: Path,
    identities: Sequence[FileIdentity],
    *,
    authorization: PredataAuthorization,
) -> None:
    """Validate both final aliases before enabling any bound-value API."""

    authorization.assert_alias_metadata_ready()
    root = artifact_root.resolve(strict=True)
    if authorization.artifact_root is None or (
        authorization.artifact_root.resolve(strict=True) != root
    ):
        raise PredataAuthorizationError("alias closure artifact root differs")
    if tuple(identity.path for identity in identities) != tuple(
        spec.final_relative_path for spec in ALIAS_SPECS
    ):
        raise IdentityError("final alias identity order or paths differ")
    for identity, spec in zip(identities, ALIAS_SPECS, strict=True):
        if (
            identity.size_bytes != spec.source_identity.size_bytes
            or identity.sha256 != spec.source_identity.sha256
        ):
            raise IdentityError("final alias byte identity differs from source")
        final = revalidate_file_identity(root, identity)
        source = revalidate_file_identity(root, spec.source_identity)
        source_stat = source.stat()
        final_stat = final.stat()
        if os.path.samefile(source, final):
            raise IdentityError("final alias is persistently linked to source")
        if (
            source_stat.st_ino
            and final_stat.st_ino
            and source_stat.st_ino == final_stat.st_ino
        ):
            raise IdentityError("final alias and source file IDs are equal")
        if source_stat.st_nlink != 1 or final_stat.st_nlink != 1:
            raise IdentityError("final alias persistent link count differs")
    census_path = root / ALIAS_SPECS[0].final_relative_path
    provenance_path = root / ALIAS_SPECS[1].final_relative_path
    validate_census_alias_document(root, census_path)
    _validate_provenance_alias_schema(
        authorization,
        provenance_path,
        metadata_only_before_handoff=True,
    )
    authorization.alias_metadata_validated = True
    authorization.validation = dataclasses.replace(
        authorization.validation, aliases_pass=True
    )


EXPECTED_PACKAGE_VERSIONS: Final[Mapping[str, str]] = {
    "joblib": "1.5.3",
    "numpy": "2.5.1",
    "pandas": "3.0.5",
    "pyarrow": "22.0.0",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.0",
    "threadpoolctl": "3.6.0",
}
EXPECTED_SKLEARN_CONFIG: Final[Mapping[str, Any]] = {
    "array_api_dispatch": False,
    "assume_finite": False,
    "display": "diagram",
    "enable_cython_pairwise_dist": True,
    "enable_metadata_routing": False,
    "pairwise_dist_chunk_size": 256,
    "print_changed_only": True,
    "skip_parameter_validation": False,
    "sparse_interface": "spmatrix",
    "transform_output": "default",
    "working_memory": 1024,
}


@dataclasses.dataclass(frozen=True, slots=True)
class AbsoluteFileIdentity:
    path: str
    size_bytes: int
    sha256: str


NATIVE_LIBRARY_IDENTITIES: Final[Mapping[str, AbsoluteFileIdentity]] = {
    "numpy_openblas": AbsoluteFileIdentity(
        (
            ".venv/Lib/"
            "site-packages/numpy.libs/"
            "libscipy_openblas64_-b788215d9d47792bcba3a2e2a7114320.dll"
        ),
        20_405_760,
        "b788215d9d47792bcba3a2e2a71143205a57282828a483f1fb071ca2c159f616",
    ),
    "scipy_openblas": AbsoluteFileIdentity(
        (
            ".venv/Lib/"
            "site-packages/scipy.libs/"
            "libscipy_openblas-197ee2fc9b4d071f7e048078cac74115.dll"
        ),
        20_262_912,
        "197ee2fc9b4d071f7e048078cac741151a6585d871061cf5f88c0c73e31963a5",
    ),
    "sklearn_vcomp140": AbsoluteFileIdentity(
        (
            ".venv/Lib/"
            "site-packages/sklearn/.libs/vcomp140.dll"
        ),
        213_072,
        "f96f3a14d88d8846f31f3ab38a490304ce7d6e4f70fae4304c63e59c7aea2d30",
    ),
}
THREADPOOL_METADATA_EXACT: Final[Mapping[str, Mapping[str, Any]]] = {
    "numpy_openblas": {
        "user_api": "blas",
        "internal_api": "openblas",
        "prefix": "libscipy_openblas",
        "version": "0.3.33.112.0",
        "threading_layer": "pthreads",
        "architecture": "Haswell",
    },
    "scipy_openblas": {
        "user_api": "blas",
        "internal_api": "openblas",
        "prefix": "libscipy_openblas",
        "version": "0.3.31.dev",
        "threading_layer": "pthreads",
        "architecture": "Haswell",
    },
    "sklearn_vcomp140": {
        "user_api": "openmp",
        "internal_api": "openmp",
        "prefix": "vcomp",
        "version": None,
        "threading_layer": None,
        "architecture": None,
    },
}
THREADPOOL_PHASES: Final[tuple[str, ...]] = (
    "BEFORE_EXPLICIT_CONTEXT",
    "INSIDE_CONTEXT",
    "AFTER_CONTEXT",
)


def _normalized_absolute_path(path: str | os.PathLike[str]) -> str:
    return Path(path).resolve(strict=True).as_posix()


def validate_pyarrow_write_table_signature(write_table: Any) -> None:
    import inspect

    parameters = inspect.signature(write_table).parameters
    expected_names = {
        "table",
        "where",
        *PARQUET_WRITE_TABLE_KWARGS_EXACT,
        *PYARROW_WRITE_TABLE_UNLISTED_DEFAULTS_EXACT,
        "kwargs",
    }
    if set(parameters) != expected_names:
        raise PredataAuthorizationError("PyArrow write_table signature keys differ")
    if (
        parameters["table"].default is not inspect.Parameter.empty
        or parameters["where"].default is not inspect.Parameter.empty
        or parameters["kwargs"].kind is not inspect.Parameter.VAR_KEYWORD
    ):
        raise PredataAuthorizationError("PyArrow write_table required parameters differ")
    for name in PARQUET_WRITE_TABLE_KWARGS_EXACT:
        if parameters[name].kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            raise PredataAuthorizationError(
                f"PyArrow explicit writer parameter kind differs: {name}"
            )
    for name, expected_default in PYARROW_WRITE_TABLE_UNLISTED_DEFAULTS_EXACT.items():
        if parameters[name].default != expected_default:
            raise PredataAuthorizationError(
                f"PyArrow unlisted writer default differs: {name}"
            )


def validate_runtime_identity() -> dict[str, Any]:
    """Validate executable, packages, sklearn config, and native DLL bytes."""

    from importlib.metadata import version

    assert_runtime_environment_caps(os.environ)
    _assert_no_linklike_absolute(Path(sys.executable), "Python executable")
    for name, identity in NATIVE_LIBRARY_IDENTITIES.items():
        _assert_no_linklike_absolute(
            Path(identity.path), f"native DLL {name}"
        )
    import joblib
    import numpy
    import pandas
    import pyarrow
    import pyarrow.parquet as pyarrow_parquet
    import scipy
    import sklearn
    import threadpoolctl
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    validate_pyarrow_write_table_signature(pyarrow_parquet.write_table)

    del (
        joblib,
        numpy,
        pandas,
        pyarrow,
        pyarrow_parquet,
        scipy,
        threadpoolctl,
        ExtraTreesRegressor,
        Ridge,
        Pipeline,
        StandardScaler,
    )
    if sys.version_info[:3] != (3, 13, 2):
        raise PredataAuthorizationError("Python version differs from 3.13.2")
    expected_executable = (
        ""
        ".venv/Scripts/python.exe"
    )
    if _normalized_absolute_path(sys.executable) != expected_executable:
        raise PredataAuthorizationError("Python executable path differs")
    observed_versions = {
        distribution: version(distribution)
        for distribution in EXPECTED_PACKAGE_VERSIONS
    }
    if observed_versions != dict(EXPECTED_PACKAGE_VERSIONS):
        raise PredataAuthorizationError(
            f"package version drift: {observed_versions!r}"
        )
    observed_config = sklearn.get_config()
    if observed_config != dict(EXPECTED_SKLEARN_CONFIG):
        raise PredataAuthorizationError(
            f"sklearn config drift: {observed_config!r}"
        )
    native: dict[str, Any] = {}
    for name, identity in NATIVE_LIBRARY_IDENTITIES.items():
        path = Path(identity.path)
        if _normalized_absolute_path(path) != identity.path:
            raise IdentityError(f"native DLL path differs: {name}")
        size, digest = _sha256_and_size(path)
        if size != identity.size_bytes or digest != identity.sha256:
            raise IdentityError(f"native DLL identity differs: {name}")
        native[name] = {
            "path": identity.path,
            "size_bytes": size,
            "sha256": digest,
        }
    threadpool_before = capture_threadpool_info_exact(expected_threads=1)
    return {
        "python_executable": expected_executable,
        "python_version": "3.13.2",
        "packages": observed_versions,
        "sklearn_config": observed_config,
        "native_libraries": native,
        "threadpool_before_explicit_context": threadpool_before,
    }


def capture_threadpool_info_exact(*, expected_threads: int) -> list[dict[str, Any]]:
    from threadpoolctl import threadpool_info

    records = threadpool_info()
    expected_paths = {
        identity.path: name for name, identity in NATIVE_LIBRARY_IDENTITIES.items()
    }
    observed_paths: set[str] = set()
    normalized_by_name: dict[str, dict[str, Any]] = {}
    for record in records:
        path = _normalized_absolute_path(record["filepath"])
        if path not in expected_paths:
            raise PredataAuthorizationError(
                f"unexpected threadpool library: {path}"
            )
        if int(record["num_threads"]) != expected_threads:
            raise PredataAuthorizationError(
                f"threadpool count differs for {path}: {record['num_threads']}"
            )
        observed_paths.add(path)
        name = expected_paths[path]
        metadata = {
            "user_api": record.get("user_api"),
            "internal_api": record.get("internal_api"),
            "prefix": record.get("prefix"),
            "version": record.get("version"),
            "threading_layer": record.get("threading_layer"),
            "architecture": record.get("architecture"),
        }
        if metadata != dict(THREADPOOL_METADATA_EXACT[name]):
            raise PredataAuthorizationError(
                f"threadpool metadata differs for {name}: {metadata!r}"
            )
        if name in normalized_by_name:
            raise PredataAuthorizationError(
                f"duplicate threadpool library record: {name}"
            )
        normalized_by_name[name] = {
            "name": name,
            "path": path,
            **metadata,
            "num_threads": int(record["num_threads"]),
        }
    if observed_paths != set(expected_paths):
        raise PredataAuthorizationError(
            "required numpy/scipy/vcomp threadpool library set differs"
        )
    return [normalized_by_name[name] for name in NATIVE_LIBRARY_IDENTITIES]


@dataclasses.dataclass(slots=True)
class ThreadpoolCaptureRecorder:
    event_count: int = 0
    phase_counts: dict[str, int] = dataclasses.field(
        default_factory=lambda: {phase: 0 for phase in THREADPOOL_PHASES}
    )
    first_by_phase: dict[str, Any] = dataclasses.field(default_factory=dict)
    last_by_phase: dict[str, Any] = dataclasses.field(default_factory=dict)
    _chain: Any = dataclasses.field(default_factory=hashlib.sha256)

    def capture(
        self,
        *,
        phase: str,
        label: str,
        observed: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if phase not in THREADPOOL_PHASES or not label:
            raise PredataAuthorizationError("threadpool capture phase differs")
        info = (
            [dict(item) for item in observed]
            if observed is not None
            else capture_threadpool_info_exact(expected_threads=1)
        )
        if len(info) != len(NATIVE_LIBRARY_IDENTITIES) or [
            item.get("name") for item in info
        ] != list(NATIVE_LIBRARY_IDENTITIES):
            raise PredataAuthorizationError(
                "supplied threadpool observation order differs"
            )
        if any(item.get("num_threads") != 1 for item in info):
            raise PredataAuthorizationError(
                "supplied threadpool observation count differs"
            )
        self.event_count += 1
        self.phase_counts[phase] += 1
        event = {
            "event_ordinal": self.event_count,
            "phase": phase,
            "label": label,
            "threadpools": info,
        }
        encoded = strict_json_dumps(event, pretty=False)
        self._chain.update(len(encoded).to_bytes(8, "big"))
        self._chain.update(encoded)
        self.first_by_phase.setdefault(phase, event)
        self.last_by_phase[phase] = event

    def snapshot(self) -> dict[str, Any]:
        if self.phase_counts["BEFORE_EXPLICIT_CONTEXT"] != 1:
            raise PredataAuthorizationError(
                "threadpool before-context capture count differs"
            )
        if self.phase_counts["INSIDE_CONTEXT"] != self.phase_counts[
            "AFTER_CONTEXT"
        ]:
            raise PredataAuthorizationError(
                "threadpool inside/after capture count differs"
            )
        required_observed = {"BEFORE_EXPLICIT_CONTEXT"}
        if self.phase_counts["INSIDE_CONTEXT"]:
            required_observed.update({"INSIDE_CONTEXT", "AFTER_CONTEXT"})
        if set(self.first_by_phase) != required_observed or set(
            self.last_by_phase
        ) != required_observed:
            raise PredataAuthorizationError(
                "threadpool capture phase coverage differs"
            )
        return {
            "capture_phases": list(THREADPOOL_PHASES),
            "capture_counts": {
                phase: self.phase_counts[phase] for phase in THREADPOOL_PHASES
            },
            "event_count": self.event_count,
            "event_chain_sha256": self._chain.hexdigest(),
            "first_observation_by_phase": {
                phase: self.first_by_phase.get(phase)
                for phase in THREADPOOL_PHASES
            },
            "last_observation_by_phase": {
                phase: self.last_by_phase.get(phase)
                for phase in THREADPOOL_PHASES
            },
        }


THREADPOOL_EVIDENCE_KEYS: Final[tuple[str, ...]] = (
    "capture_phases",
    "capture_counts",
    "event_count",
    "event_chain_sha256",
    "first_observation_by_phase",
    "last_observation_by_phase",
)
THREADPOOL_EVENT_KEYS: Final[tuple[str, ...]] = (
    "event_ordinal",
    "phase",
    "label",
    "threadpools",
)
THREADPOOL_OBSERVATION_KEYS: Final[tuple[str, ...]] = (
    "name",
    "path",
    "user_api",
    "internal_api",
    "prefix",
    "version",
    "threading_layer",
    "architecture",
    "num_threads",
)


def _validate_persisted_threadpool_observation(value: Any) -> None:
    if not isinstance(value, list) or len(value) != len(
        NATIVE_LIBRARY_IDENTITIES
    ):
        raise IntegrityError("persisted threadpool observation count differs")
    for item, name in zip(value, NATIVE_LIBRARY_IDENTITIES, strict=True):
        record = _require_exact_keys(
            item, THREADPOOL_OBSERVATION_KEYS, "threadpool observation"
        )
        identity = NATIVE_LIBRARY_IDENTITIES[name]
        if (
            record["name"] != name
            or record["path"] != identity.path
            or record["num_threads"] != 1
            or {
                key: record[key]
                for key in THREADPOOL_METADATA_EXACT[name]
            }
            != dict(THREADPOOL_METADATA_EXACT[name])
        ):
            raise IntegrityError("persisted threadpool metadata differs")


def validate_threadpool_evidence(
    evidence: Mapping[str, Any], fit_ledger: Mapping[str, Any]
) -> None:
    record = _require_exact_keys(
        evidence, THREADPOOL_EVIDENCE_KEYS, "threadpool evidence"
    )
    validate_fit_ledger_document(fit_ledger)
    if record["capture_phases"] != list(THREADPOOL_PHASES):
        raise IntegrityError("threadpool phase order differs")
    counts = _require_exact_keys(
        record["capture_counts"], THREADPOOL_PHASES, "threadpool counts"
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counts.values()
    ):
        raise IntegrityError("threadpool count type differs")
    executed = fit_ledger["executed_counts"]
    expected_inside = (
        2 * int(executed["completed_primary_model_units"])
        + int(executed["completed_analytic_affine_units"])
    )
    if counts != {
        "BEFORE_EXPLICIT_CONTEXT": 1,
        "INSIDE_CONTEXT": expected_inside,
        "AFTER_CONTEXT": expected_inside,
    }:
        raise IntegrityError("threadpool fit-ledger count crosslink differs")
    if (
        not isinstance(record["event_count"], int)
        or isinstance(record["event_count"], bool)
        or record["event_count"] != 1 + 2 * expected_inside
        or not isinstance(record["event_chain_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", record["event_chain_sha256"])
        is None
    ):
        raise IntegrityError("threadpool event aggregate differs")
    for boundary_key in (
        "first_observation_by_phase",
        "last_observation_by_phase",
    ):
        boundaries = _require_exact_keys(
            record[boundary_key], THREADPOOL_PHASES, boundary_key
        )
        for phase in THREADPOOL_PHASES:
            event = boundaries[phase]
            if phase != "BEFORE_EXPLICIT_CONTEXT" and expected_inside == 0:
                if event is not None:
                    raise IntegrityError("zero-fit threadpool boundary is nonnull")
                continue
            item = _require_exact_keys(
                event, THREADPOOL_EVENT_KEYS, "threadpool boundary event"
            )
            if (
                item["phase"] != phase
                or not isinstance(item["event_ordinal"], int)
                or isinstance(item["event_ordinal"], bool)
                or item["event_ordinal"] <= 0
                or not isinstance(item["label"], str)
                or not item["label"]
            ):
                raise IntegrityError("threadpool boundary event differs")
            _validate_persisted_threadpool_observation(item["threadpools"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sealed target-free duplicate V5 diagnostic"
    )
    # Keep raw path lexemes until production path authorization has completed.
    parser.add_argument("--root", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--independent-go", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    control_validator: ControlValidator | None = None,
    actual_orig_argv_for_test: list[str] | None = None,
) -> int:
    """Run the diagnostic; explicit ``argv`` is a non-authority test seam."""

    production_invocation = argv is None
    args = _build_parser().parse_args(argv)
    if production_invocation:
        artifact_root, authorization_path, independent_go_path = (
            validate_production_cli_paths(
                args.root, args.authorization, args.independent_go
            )
        )
        actual_orig_argv: Any = getattr(sys, "orig_argv", None)
    else:
        artifact_root = Path(args.root).resolve(strict=True)
        authorization_path = Path(args.authorization)
        independent_go_path = Path(args.independent_go)
        actual_orig_argv = actual_orig_argv_for_test
    if control_validator is not None:
        supplemental = PredataAuthorization(control_validator(artifact_root))
        supplemental.assert_ready()
    prepare_kwargs: dict[str, Any] = {
        "authorization_path": authorization_path,
        "independent_go_path": independent_go_path,
    }
    if production_invocation or actual_orig_argv is not None:
        prepare_kwargs["actual_orig_argv"] = actual_orig_argv
        prepare_kwargs["require_actual_orig_argv"] = True
    prepared = prepare_execution(artifact_root, **prepare_kwargs)
    result = run_prepared_execution(prepared)
    payload = strict_json_dumps(result.stdout_document(), pretty=True)
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

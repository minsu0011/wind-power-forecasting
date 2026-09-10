"""Fail-closed provenance audit primitives for the champion external NWP.

The Open-Meteo Previous Runs API names fixed lead-time offsets with suffixes
such as ``previous_day1``.  A suffix is not an issuance or publication
receipt.  This module therefore requires an exact run identifier, issue time,
publication/availability time, and a verified immutable raw payload before a
row can be called cutoff compliant.

Only external weather artifacts are read here.  Labels, public feedback,
models, predictions, and submission values are outside this module's scope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
UTC = ZoneInfo("UTC")
CUTOFF_HOUR_KST = 14
SOURCE_ORDER = ("ecmwf_ifs025", "icon_global", "gfs_global")
GROUP_ORDER = ("kpx_group_1", "kpx_group_2", "kpx_group_3")

SOURCE_CONTRACT: dict[str, dict[str, str]] = {
    "ecmwf_ifs025": {
        "short_id": "ecmwf",
        "upstream_provider": "ECMWF",
        "product": "IFS 0.25 degree",
        "history_manifest": (
            "artifacts/external/openmeteo_ecmwf_ifs025_previous_runs_v1/"
            "source_manifest.json"
        ),
        "history_downloader": "scripts/download_openmeteo_ecmwf_previous_runs.py",
        "vertical_method": "native 100 m wind",
        "official_model_url": "https://open-meteo.com/en/docs/ecmwf-api",
    },
    "icon_global": {
        "short_id": "icon",
        "upstream_provider": "DWD",
        "product": "ICON Global",
        "history_manifest": (
            "artifacts/external/openmeteo_icon_global_previous_runs_v1/"
            "source_manifest.json"
        ),
        "history_downloader": (
            "scripts/download_openmeteo_icon_global_previous_runs.py"
        ),
        "vertical_method": (
            "100 m vector midpoint of native 80 m and 120 m vectors"
        ),
        "official_model_url": "https://open-meteo.com/en/docs/dwd-api",
    },
    "gfs_global": {
        "short_id": "gfs",
        "upstream_provider": "NOAA NCEP",
        "product": "GFS Global",
        "history_manifest": (
            "artifacts/external/openmeteo_gfs_global_previous_runs_v1/"
            "source_manifest.json"
        ),
        "history_downloader": (
            "scripts/download_openmeteo_gfs_revision_previous_runs.py"
        ),
        "vertical_method": "native 100 m wind; native 80 m used for shear",
        "official_model_url": "https://open-meteo.com/en/docs/gfs-api",
    },
}

FINAL_MANIFEST = (
    "artifacts/external/openmeteo_multi_nwp_g23_rescue_2025_v1/"
    "source_manifest.json"
)
FINAL_DOWNLOADER = "scripts/download_openmeteo_multi_nwp_g23_rescue_2025.py"
PREVIOUS_RUNS_URL = "https://open-meteo.com/en/docs/previous-runs-api"
LICENSE_URL = "https://open-meteo.com/en/license"
TERMS_URL = "https://open-meteo.com/en/terms"


class ProvenanceFailure(AssertionError):
    """Raised when an artifact identity or schema is inconsistent."""


@dataclass(frozen=True)
class CutoffAssessment:
    valid_time_kst: str
    target_day_cutoff_kst: str
    previous_day_offset: int
    nominal_fixed_lead_reference_kst: str
    nominal_reference_before_cutoff: bool
    exact_run_identifier_present: bool
    exact_issue_time_present: bool
    exact_publication_time_present: bool
    immutable_raw_sha_verified: bool
    issue_before_cutoff: bool | None
    publication_before_cutoff: bool | None
    cutoff_compliant_proven: bool
    reason: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _declared_size(record: Mapping[str, Any]) -> int:
    for key in ("size_bytes", "bytes"):
        if key in record:
            return int(record[key])
    raise ProvenanceFailure("declared artifact has no byte count")


def verify_declared_file(record: Mapping[str, Any], root: Path) -> Path:
    raw = Path(str(record.get("path", "")))
    path = raw if raw.is_absolute() else root / raw
    path = path.resolve()
    if not path.is_file():
        raise ProvenanceFailure(f"declared artifact missing: {path}")
    if path.stat().st_size != _declared_size(record):
        raise ProvenanceFailure(f"declared artifact size changed: {path}")
    if sha256_file(path) != str(record.get("sha256")):
        raise ProvenanceFailure(f"declared artifact hash changed: {path}")
    return path


def selected_previous_day(valid_time: pd.Timestamp | datetime) -> int:
    """Mirror the champion's frozen hour mask without claiming provenance."""

    timestamp = pd.Timestamp(valid_time)
    return 1 if 1 <= int(timestamp.hour) <= 13 else 2


def _as_kst(value: str | datetime | pd.Timestamp) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(KST)
    else:
        timestamp = timestamp.tz_convert(KST)
    return timestamp.to_pydatetime()


def assess_cutoff_evidence(
    valid_time: str | datetime | pd.Timestamp,
    previous_day_offset: int,
    *,
    exact_run_identifier: str | None,
    issue_time: str | datetime | pd.Timestamp | None,
    publication_time: str | datetime | pd.Timestamp | None,
    immutable_raw_sha_verified: bool,
) -> CutoffAssessment:
    """Assess one fixed-lead value against D-1 14:00 KST, fail closed."""

    if previous_day_offset not in (1, 2):
        raise ValueError("champion cutoff audit accepts only previous_day1/day2")
    valid = _as_kst(valid_time)
    cutoff = (valid.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)).replace(
        hour=CUTOFF_HOUR_KST
    )
    nominal = valid - timedelta(days=previous_day_offset)
    issue = _as_kst(issue_time) if issue_time is not None else None
    publication = (
        _as_kst(publication_time) if publication_time is not None else None
    )
    issue_before = issue <= cutoff if issue is not None else None
    publication_before = publication <= cutoff if publication is not None else None
    exact_fields = bool(exact_run_identifier) and issue is not None and publication is not None
    passed = bool(
        exact_fields
        and immutable_raw_sha_verified
        and issue_before is True
        and publication_before is True
    )
    if passed:
        reason = "exact run, issue, publication, cutoff, and immutable raw proof complete"
    else:
        missing: list[str] = []
        if not exact_run_identifier:
            missing.append("exact_run_identifier")
        if issue is None:
            missing.append("issue_time")
        if publication is None:
            missing.append("publication_time")
        if not immutable_raw_sha_verified:
            missing.append("immutable_raw_sha")
        if issue_before is False:
            missing.append("issue_after_cutoff")
        if publication_before is False:
            missing.append("publication_after_cutoff")
        reason = "missing_or_failed:" + ",".join(missing)
    return CutoffAssessment(
        valid_time_kst=valid.isoformat(),
        target_day_cutoff_kst=cutoff.isoformat(),
        previous_day_offset=previous_day_offset,
        nominal_fixed_lead_reference_kst=nominal.isoformat(),
        nominal_reference_before_cutoff=nominal <= cutoff,
        exact_run_identifier_present=bool(exact_run_identifier),
        exact_issue_time_present=issue is not None,
        exact_publication_time_present=publication is not None,
        immutable_raw_sha_verified=bool(immutable_raw_sha_verified),
        issue_before_cutoff=issue_before,
        publication_before_cutoff=publication_before,
        cutoff_compliant_proven=passed,
        reason=reason,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProvenanceFailure(f"manifest is not an object: {path}")
    return value


def _to_kst_text(value: str | None) -> str | None:
    if not value:
        return None
    return _as_kst(value).isoformat()


def _raw_identity(
    request: Mapping[str, Any], root: Path
) -> tuple[str | None, int | None, str | None, bool, bool, list[str]]:
    record = request.get("raw_response") or request.get("response")
    if isinstance(record, Mapping):
        path = verify_declared_file(record, root)
        payload_keys: list[str] = []
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                payload_keys = sorted(map(str, payload.keys()))
        return (
            str(path),
            _declared_size(record),
            str(record["sha256"]),
            True,
            True,
            payload_keys,
        )
    digest = request.get("response_sha256")
    size = request.get("response_bytes")
    return (
        None,
        int(size) if size is not None else None,
        str(digest) if digest is not None else None,
        False,
        False,
        [],
    )


def _request_parameters(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "host": parsed.netloc,
        "path": parsed.path,
        "query": {
            key: values if len(values) != 1 else values[0]
            for key, values in sorted(parse_qs(parsed.query).items())
        },
    }


def _coverage(
    frame: pd.DataFrame, group: str, variables: list[str]
) -> dict[str, Any]:
    subset = frame.loc[frame["group"].eq(group), ["time", *variables]].copy()
    if subset.empty:
        raise ProvenanceFailure(f"normalized source lacks group: {group}")
    subset["time"] = pd.to_datetime(subset["time"])
    if subset["time"].duplicated().any():
        raise ProvenanceFailure(f"normalized source has duplicate time: {group}")
    complete = subset[variables].notna().all(axis=1)
    return {
        "rows": int(len(subset)),
        "complete_rows": int(complete.sum()),
        "first_time": subset["time"].min().isoformat(),
        "last_time": subset["time"].max().isoformat(),
        "first_complete_time": (
            subset.loc[complete, "time"].min().isoformat() if complete.any() else None
        ),
        "last_complete_time": (
            subset.loc[complete, "time"].max().isoformat() if complete.any() else None
        ),
        "missing_value_cells": int(subset[variables].isna().sum().sum()),
    }


def _period_rows(
    *,
    root: Path,
    source_id: str,
    period: str,
    calendar_year: int,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    source_section: Mapping[str, Any],
    downloader_path: Path,
) -> list[dict[str, Any]]:
    contract = SOURCE_CONTRACT[source_id]
    normalized_record = source_section.get("normalized")
    if not isinstance(normalized_record, Mapping):
        raise ProvenanceFailure(f"{source_id}/{period}: normalized identity absent")
    normalized_path = verify_declared_file(normalized_record, root)
    frame = pd.read_parquet(normalized_path)
    variables = list(source_section.get("variables") or manifest.get("variables") or [])
    if not variables or not {"group", "time", *variables}.issubset(frame.columns):
        raise ProvenanceFailure(f"{source_id}/{period}: normalized schema differs")
    requests = source_section.get("requests")
    if not isinstance(requests, list) or len(requests) != 3:
        raise ProvenanceFailure(f"{source_id}/{period}: request cardinality differs")
    by_group = {
        str(item.get("group")): item
        for item in requests
        if isinstance(item, Mapping)
    }
    if list(by_group) != list(GROUP_ORDER):
        raise ProvenanceFailure(f"{source_id}/{period}: request group order differs")
    manifest_time = manifest.get("retrieved_utc") or manifest.get("created_utc")
    downloader = file_record(downloader_path)
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        request = by_group[group]
        url = str(request.get("url", ""))
        parameters = _request_parameters(url)
        query = parameters["query"]
        if (
            parameters["host"] != "previous-runs-api.open-meteo.com"
            or query.get("models") != source_id
            or query.get("timezone") != "Asia/Seoul"
            or query.get("cell_selection") != "nearest"
        ):
            raise ProvenanceFailure(f"{source_id}/{period}/{group}: URL differs")
        raw_path, raw_bytes, raw_sha, raw_present, raw_verified, raw_keys = (
            _raw_identity(request, root)
        )
        coverage = _coverage(frame, group, variables)
        representative = coverage["first_complete_time"] or coverage["first_time"]
        assessment = assess_cutoff_evidence(
            representative,
            selected_previous_day(pd.Timestamp(representative)),
            exact_run_identifier=None,
            issue_time=None,
            publication_time=None,
            immutable_raw_sha_verified=raw_verified,
        )
        returned = request.get("returned_grid") or request.get("coverage") or {}
        if not isinstance(returned, Mapping):
            returned = {}
        requested = request.get("requested_coordinate") or {
            "latitude": query.get("latitude"),
            "longitude": query.get("longitude"),
        }
        if not isinstance(requested, Mapping):
            requested = {}
        rows.append(
            {
                "source_id": source_id,
                "source_short_id": contract["short_id"],
                "upstream_provider": contract["upstream_provider"],
                "upstream_product_model": contract["product"],
                "archive_provider": "Open-Meteo",
                "archive_product": "Previous Model Runs API fixed lead offsets",
                "archive_period": period,
                "calendar_year": calendar_year,
                "group": group,
                "champion_group_used": group in ("kpx_group_1", "kpx_group_2"),
                "champion_consumers": "multi_nwp_joint_g12;multi_nwp_agreement_gate_g12",
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "downloader_path": downloader["path"],
                "downloader_sha256": downloader["sha256"],
                "endpoint": str(manifest.get("endpoint")),
                "request_url": url,
                "request_parameters_json": json.dumps(
                    parameters["query"], sort_keys=True, separators=(",", ":")
                ),
                "request_model_parameter": str(query.get("models")),
                "request_has_run_parameter": "run" in query,
                "request_has_initialization_parameter": any(
                    key in query for key in ("run", "initialization", "issue_time")
                ),
                "download_time_utc": str(manifest_time) if manifest_time else None,
                "download_time_kst": _to_kst_text(
                    str(manifest_time) if manifest_time else None
                ),
                "download_time_scope": (
                    "batch_manifest_completion_not_per_request"
                    if manifest_time
                    else "missing"
                ),
                "target_valid_start_kst": coverage["first_time"],
                "target_valid_end_kst": coverage["last_time"],
                "first_complete_time_kst": coverage["first_complete_time"],
                "last_complete_time_kst": coverage["last_complete_time"],
                "normalized_rows": coverage["rows"],
                "complete_rows": coverage["complete_rows"],
                "missing_value_cells": coverage["missing_value_cells"],
                "variables_json": json.dumps(variables, separators=(",", ":")),
                "cutoff_rule": "target day D: all values public/fixed by D-1 14:00 KST",
                "selection_mask": "hours 01-13 previous_day1; 00,14-23 previous_day2",
                "previous_day_is_only_fixed_lead_offset": True,
                "exact_run_identifier_present": False,
                "exact_issue_time_present": False,
                "exact_publication_time_present": False,
                "per_row_availability_receipt_present": False,
                "cutoff_proven_rows": 0,
                "cutoff_compliant_proven": assessment.cutoff_compliant_proven,
                "cutoff_failure_reason": (
                    "Previous Runs fields and raw responses contain valid-time values but "
                    "no exact run identifier, initialization/issue time, or publication/"
                    "availability time. Nominal 24/48-hour offset is insufficient."
                ),
                "representative_nominal_reference_before_cutoff": (
                    assessment.nominal_reference_before_cutoff
                ),
                "raw_artifact_path": raw_path,
                "raw_declared_bytes": raw_bytes,
                "raw_declared_sha256": raw_sha,
                "raw_file_present": raw_present,
                "raw_sha256_verified": raw_verified,
                "raw_response_top_level_keys_json": json.dumps(raw_keys),
                "raw_has_model_run_metadata": any(
                    key in raw_keys
                    for key in (
                        "run",
                        "run_time",
                        "initialization_time",
                        "issue_time",
                        "publication_time",
                        "available_at",
                    )
                ),
                "normalized_path": str(normalized_path),
                "normalized_bytes": normalized_path.stat().st_size,
                "normalized_sha256": sha256_file(normalized_path),
                "normalized_sha256_verified": True,
                "requested_latitude": str(requested.get("latitude")),
                "requested_longitude": str(requested.get("longitude")),
                "returned_latitude": str(
                    returned.get("latitude", returned.get("returned_latitude"))
                ),
                "returned_longitude": str(
                    returned.get("longitude", returned.get("returned_longitude"))
                ),
                "spatial_crosswalk": "KPX group arithmetic centroid; nearest model grid cell",
                "crosswalk_source_info_xlsx_identity_bound_in_source_manifest": False,
                "interpolation_method": (
                    "Open-Meteo hourly interpolation plus " + contract["vertical_method"]
                ),
                "public_request_url_present": bool(url),
                "historical_http_receipt_or_availability_header_bound": False,
                "license_url": LICENSE_URL,
                "terms_url": TERMS_URL,
                "license_identified_current_official_docs": (
                    "CC BY 4.0; attribution and modification notice required"
                ),
                "license_snapshot_bound_at_download": False,
                "attribution_record_bound_in_source_or_champion_manifest": False,
                "same_pipeline_2022_through_2025_proven": False,
                "post_cutoff_or_reanalysis_exclusion_proven": False,
                "row_verdict": "STOP_PROVENANCE_FAILURE",
            }
        )
    return rows


def build_champion_provenance_rows(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify local identities and build the 18-row provenance ledger."""

    root = root.resolve()
    final_manifest_path = root / FINAL_MANIFEST
    final_manifest = _load_manifest(final_manifest_path)
    final_sources = final_manifest.get("sources")
    if not isinstance(final_sources, Mapping) or set(final_sources) != set(SOURCE_ORDER):
        raise ProvenanceFailure("2025 source manifest source order differs")
    rows: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = [file_record(final_manifest_path)]
    final_downloader = root / FINAL_DOWNLOADER
    input_records.append(file_record(final_downloader))
    for source_id in SOURCE_ORDER:
        contract = SOURCE_CONTRACT[source_id]
        history_path = root / contract["history_manifest"]
        history = _load_manifest(history_path)
        history_downloader = root / contract["history_downloader"]
        input_records.extend(
            [file_record(history_path), file_record(history_downloader)]
        )
        rows.extend(
            _period_rows(
                root=root,
                source_id=source_id,
                period="historical_training_2024",
                calendar_year=2024,
                manifest_path=history_path,
                manifest=history,
                source_section=history,
                downloader_path=history_downloader,
            )
        )
        source_section = final_sources[source_id]
        if not isinstance(source_section, Mapping):
            raise ProvenanceFailure(f"2025 source section differs: {source_id}")
        rows.extend(
            _period_rows(
                root=root,
                source_id=source_id,
                period="final_application_2025",
                calendar_year=2025,
                manifest_path=final_manifest_path,
                manifest=final_manifest,
                source_section=source_section,
                downloader_path=final_downloader,
            )
        )
    frame = pd.DataFrame(rows)
    expected_keys = [
        (source, period, group)
        for source in SOURCE_ORDER
        for period in ("historical_training_2024", "final_application_2025")
        for group in GROUP_ORDER
    ]
    observed_keys = list(
        frame[["source_id", "archive_period", "group"]].itertuples(
            index=False, name=None
        )
    )
    if observed_keys != expected_keys:
        raise ProvenanceFailure("provenance row order/cardinality differs")
    if frame["cutoff_compliant_proven"].any():
        raise ProvenanceFailure("missing publication evidence was promoted to PASS")
    metadata = {
        "input_records": input_records,
        "source_order": list(SOURCE_ORDER),
        "group_order": list(GROUP_ORDER),
        "row_order_sha256": hashlib.sha256(
            ("\n".join("|".join(key) for key in expected_keys) + "\n").encode()
        ).hexdigest(),
        "rows": len(frame),
    }
    return frame, metadata


def provenance_audit_payload(
    *,
    root: Path,
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
    prompt_record: Mapping[str, Any],
    parquet_record: Mapping[str, Any],
) -> dict[str, Any]:
    source_summaries: dict[str, Any] = {}
    for source_id in SOURCE_ORDER:
        source = frame.loc[frame["source_id"].eq(source_id)]
        source_summaries[source_id] = {
            "upstream_provider": SOURCE_CONTRACT[source_id]["upstream_provider"],
            "product": SOURCE_CONTRACT[source_id]["product"],
            "periods": sorted(source["archive_period"].unique().tolist()),
            "calendar_years_locally_present": sorted(
                map(int, source["calendar_year"].unique())
            ),
            "raw_files_present_rows": int(source["raw_file_present"].sum()),
            "raw_files_expected_rows": int(len(source)),
            "processed_hash_verified_rows": int(
                source["normalized_sha256_verified"].sum()
            ),
            "exact_run_issue_publication_rows": int(
                (
                    source["exact_run_identifier_present"]
                    & source["exact_issue_time_present"]
                    & source["exact_publication_time_present"]
                ).sum()
            ),
            "cutoff_proven_rows": int(source["cutoff_proven_rows"].sum()),
            "same_pipeline_2022_through_2025_proven": False,
            "verdict": "PUBLIC_SCORED_BUT_FINAL_REPRO_UNPROVEN",
        }
    checks = {
        "provider_product_model_named": True,
        "request_urls_and_parameters_bound": bool(
            frame["public_request_url_present"].all()
        ),
        "processed_parquet_hashes_verified": bool(
            frame["normalized_sha256_verified"].all()
        ),
        "all_raw_payloads_present_and_hash_verified": bool(
            frame["raw_sha256_verified"].all()
        ),
        "all_requests_have_exact_download_time": bool(
            frame["download_time_utc"].notna().all()
            and frame["download_time_scope"].eq("per_request").all()
        ),
        "exact_run_issue_time_for_every_used_value": bool(
            frame["exact_run_identifier_present"].all()
            and frame["exact_issue_time_present"].all()
        ),
        "publication_availability_by_cutoff_for_every_used_value": bool(
            frame["cutoff_compliant_proven"].all()
        ),
        "license_terms_snapshot_bound_at_download": bool(
            frame["license_snapshot_bound_at_download"].all()
        ),
        "required_attribution_bound": bool(
            frame["attribution_record_bound_in_source_or_champion_manifest"].all()
        ),
        "spatial_crosswalk_source_identity_bound": bool(
            frame["crosswalk_source_info_xlsx_identity_bound_in_source_manifest"].all()
        ),
        "same_pipeline_reproduced_2022_through_2025": bool(
            frame["same_pipeline_2022_through_2025_proven"].all()
        ),
        "post_cutoff_reanalysis_hindsight_excluded_with_exact_evidence": bool(
            frame["post_cutoff_or_reanalysis_exclusion_proven"].all()
        ),
        "previous_day_name_alone_treated_as_sufficient": False,
    }
    blocking = [
        {
            "id": "B1_NO_EXACT_RUN_OR_ISSUE_TIME",
            "evidence": (
                "All request URLs omit a run/initialization parameter; persisted raw "
                "responses expose valid-time arrays and generationtime_ms (API response "
                "processing duration), not a model run identifier or initialization time."
            ),
        },
        {
            "id": "B2_NO_PUBLICATION_AVAILABILITY_RECEIPT",
            "evidence": (
                "No per-value or per-request publication/availability timestamp or HTTP "
                "availability receipt proves that each value was public and fixed by "
                "D-1 14:00 KST. previous_day1/day2 proves only nominal fixed lead."
            ),
        },
        {
            "id": "B3_INCOMPLETE_IMMUTABLE_RAW_2024",
            "evidence": (
                "The 2024 ECMWF and ICON manifests record response hashes but retain no "
                "raw response file; only their processed Parquet files remain."
            ),
        },
        {
            "id": "B4_DOWNLOAD_TIME_NOT_REQUEST_EXACT",
            "evidence": (
                "ECMWF/ICON 2024 and unified 2025 timestamps are batch manifest creation "
                "times, not request receipts; the GFS 2024 manifest has no retrieval time."
            ),
        },
        {
            "id": "B5_2022_2025_PIPELINE_PARITY_ABSENT",
            "evidence": (
                "Local three-source material covers partial 2024 plus 2025 only; no same "
                "ECMWF/ICON/GFS pipeline is present for 2022 or 2023, and 2024/2025 use "
                "different downloaders."
            ),
        },
        {
            "id": "B6_LICENSE_ATTRIBUTION_NOT_IMMUTABLY_BOUND",
            "evidence": (
                "Current official Open-Meteo pages identify CC BY 4.0 and attribution, "
                "but source/champion manifests bind neither a contemporaneous licence "
                "snapshot nor an attribution record."
            ),
        },
    ]
    return {
        "schema_version": 1,
        "phase": "B_CHAMPION_EXTERNAL_DATA_PROVENANCE",
        "prompt": dict(prompt_record),
        "audit_scope": (
            "multi_nwp_joint_g12 and multi_nwp_agreement_gate_g12 external "
            "Open-Meteo ECMWF/ICON/GFS only"
        ),
        "claim_class": "LOCAL_REPRODUCED_PROVENANCE_AUDIT",
        "champion_public_anchor": {
            "csv": "multi_nwp_joint_g12_posthoc_rescue_recent097_2025.csv",
            "sha256": "05f0f45a38044605cc43b78a60fa4daf8373ffe47ff881839e6914c622fa1c9e",
            "score": 0.6142203,
            "note": "Public score is not used to excuse or select provenance evidence.",
        },
        "official_documentation": [
            {
                "url": PREVIOUS_RUNS_URL,
                "finding": (
                    "previous_day1/day2 are fixed lead offsets; official docs direct "
                    "workflows needing initialization datetime to the Single Runs API"
                ),
                "evidence_limit": (
                    "live 2026 reference, not an immutable 2024/2025 publication receipt"
                ),
            },
            {
                "url": LICENSE_URL,
                "finding": "API data CC BY 4.0; attribution and modification notice required",
                "evidence_limit": "not bound contemporaneously by source manifests",
            },
            {
                "url": TERMS_URL,
                "finding": "free API use is non-commercial and data availability is not guaranteed",
                "evidence_limit": "not bound contemporaneously by source manifests",
            },
        ],
        "source_summaries": source_summaries,
        "checks": checks,
        "blocking_failures": blocking,
        "previous_day_name_policy": (
            "A variable suffix is never sufficient. Exact run ID, issue time, public "
            "availability time, and immutable raw hash are mandatory."
        ),
        "phase_b_verdict": "PUBLIC_SCORED_BUT_FINAL_REPRO_UNPROVEN",
        "terminal_decision": "STOP_PROVENANCE_FAILURE",
        "external_nwp_final_candidate_eligibility": False,
        "provenance_table": dict(parquet_record),
        "row_order_sha256": metadata["row_order_sha256"],
        "input_records": list(metadata["input_records"]),
        "access_ledger": {
            "external_source_manifests_read": 4,
            "external_processed_parquets_read": 6,
            "external_raw_json_files_read": int(frame["raw_file_present"].sum()),
            "label_files_read": 0,
            "public_feedback_files_read": 0,
            "model_files_loaded": 0,
            "prediction_files_read": 0,
            "submission_values_read": 0,
            "network_requests_by_audit_code": 0,
        },
        "reproduction": {
            "command": (
                "python scripts/run_champion_provenance_audit.py "
                "--v2-root artifacts/baram2026_evidence_first_v2_20260810_160144"
            ),
            "no_overwrite": True,
            "root": str(root.resolve()),
        },
    }


def cutoff_assessment_dict(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """JSON-friendly convenience wrapper used by tests and reports."""

    return asdict(assess_cutoff_evidence(*args, **kwargs))

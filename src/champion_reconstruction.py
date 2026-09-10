"""Static feasibility audit for the multi-NWP G1/G2 champion analogue.

This module deliberately has no dataframe or model-library dependency.  It may
read JSON manifests and hash files, but it cannot parse prediction, feature, or
label arrays and cannot fit or score a candidate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_MANIFESTS = {
    "ecmwf": Path(
        "artifacts/external/openmeteo_ecmwf_ifs025_previous_runs_v1/source_manifest.json"
    ),
    "icon": Path(
        "artifacts/external/openmeteo_icon_global_previous_runs_v1/source_manifest.json"
    ),
    "gfs": Path(
        "artifacts/external/openmeteo_gfs_global_previous_runs_v1/source_manifest.json"
    ),
}

EXPECTED_SOURCE_MANIFEST_SHA256 = {
    "ecmwf": "93c4fe23824dd80e427fd924d27592e93e99a3ed87d7243afb44efb715040132",
    "icon": "e19eebc3dc94fda22244df986cbb313b521938b5b2ae29165c3231c7c05ba70a",
    "gfs": "53a93bcc441e5e3bdeeeade951035abef396e168e809599acc8b8e4e61bfc2bb",
}

RECIPE_IDENTITIES = {
    "feature_module": (
        Path("src/multi_nwp_joint.py"),
        "38e43101fd1af770d788e0f26e76f7d916bb6f6c95b1fde206fabe59d9524b56",
    ),
    "causal_2024_runner": (
        Path("scripts/run_multi_nwp_joint_disagreement_paired_increment.py"),
        "6191b724c4537f2273257823e2f9d5cfd7e109ac417109b8f394df80171d88a0",
    ),
    "causal_2024_preregister": (
        Path("configs/multi_nwp_joint_disagreement_paired_increment_preregister_v1.json"),
        "bf52d742cb9b0160f31d27e949f6f1b4575c40575472520e9307f5af998957af",
    ),
    "posthoc_g12_runner": (
        Path("scripts/run_multi_nwp_joint_g12_posthoc_rescue.py"),
        "2beae2ab3cc3b3c8002aca6a1d6fea12b6c2ea062aa36d96992d283479baeb8f",
    ),
    "posthoc_g12_preregister": (
        Path("configs/multi_nwp_joint_g12_posthoc_rescue_preregister_v1.json"),
        "1bbd3bb71b418eb1ca9e16e1ebf7bd6a911936b41a35c1e6ba14171a69962c30",
    ),
    "final_runner": (
        Path("scripts/run_multi_nwp_joint_g12_final.py"),
        "2d878ce113b786536dc89783f1d29013537a74a9a1e9dee54a8e13ef9f1cbca3",
    ),
    "final_preregister": (
        Path("configs/multi_nwp_joint_g12_final_preregister_v1.json"),
        "b9fc9f63147cfd518b83bc56adfa57cc328264c1dfc59cb569d7263533c74e8d",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(root: Path, relative: Path, expected: str) -> dict[str, Any]:
    path = root / relative
    observed = sha256_file(path) if path.is_file() else None
    return {
        "path": relative.as_posix(),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "expected_sha256": expected,
        "observed_sha256": observed,
        "matches_expected": observed == expected,
    }


def assess_champion_reconstruction(root: Path) -> dict[str, Any]:
    """Return a metadata-only STOP/GO decision; never open parquet/csv values."""

    root = root.resolve()
    source_ids = {
        name: _identity(root, path, EXPECTED_SOURCE_MANIFEST_SHA256[name])
        for name, path in SOURCE_MANIFESTS.items()
    }
    recipe_ids = {
        name: _identity(root, path, expected)
        for name, (path, expected) in RECIPE_IDENTITIES.items()
    }

    manifests = {
        name: _load_json(root / relative)
        for name, relative in SOURCE_MANIFESTS.items()
        if (root / relative).is_file()
    }
    icon_probes = {
        str(item.get("date"))[:4]: sum(item.get("non_null_by_variable", {}).values())
        for item in manifests.get("icon", {}).get("fixed_probes", [])
    }
    ecmwf_requests = manifests.get("ecmwf", {}).get("requests", [])
    ecmwf_years = sorted(
        {
            str(item.get("start", ""))[:4]
            for item in ecmwf_requests
            if item.get("start")
        }
    )
    if not ecmwf_years:
        first_complete = manifests.get("ecmwf", {}).get("normalized", {}).get(
            "first_complete_time"
        )
        if first_complete:
            ecmwf_years = [str(first_complete)[:4]]
    gfs_years_raw = manifests.get("gfs", {}).get(
        "calendar_years_requested_or_parsed", []
    )
    gfs_years = [str(gfs_years_raw)] if isinstance(gfs_years_raw, int) else [
        str(year) for year in gfs_years_raw
    ]

    evidence = {
        "icon_2022_probe_total_non_null": icon_probes.get("2022"),
        "icon_2023_probe_total_non_null": icon_probes.get("2023"),
        "ecmwf_request_years_in_manifest": ecmwf_years,
        "gfs_request_years_in_manifest": gfs_years,
        "pre2024_exact_three_source_bundle_present": False,
        "ecmwf_raw_response_retained": False,
        "icon_raw_response_retained": False,
        "gfs_raw_response_retained": True,
    }
    blockers = [
        {
            "id": "NO_PRE2024_THREE_SOURCE_HISTORY",
            "fatal": True,
            "detail": (
                "ICON fixed probes for 2022 and 2023 contain zero values, while "
                "the ECMWF and GFS manifests contain no 2022/2023 retrieval."
            ),
        },
        {
            "id": "G12_OWNERSHIP_POSTHOC",
            "fatal": True,
            "detail": (
                "The all-group causal experiment rejected; G1/G2-only ownership "
                "was selected after the 2024 result was opened."
            ),
        },
        {
            "id": "ONLY_2024_IS_CONTAMINATED_STRESS",
            "fatal": True,
            "detail": (
                "The sole complete same-recipe forward split is 2024 H1-to-H2, "
                "which V2 reserves for stress rather than discovery or selection."
            ),
        },
        {
            "id": "RAW_PROVENANCE_NOT_BYTE_REPLAYABLE",
            "fatal": False,
            "detail": (
                "ECMWF and ICON historical response bytes were not retained; only "
                "hash-bound normalized parquet and request metadata remain."
            ),
        },
    ]
    return {
        "schema_version": 1,
        "assessment_kind": "static_metadata_only_no_label_or_array_access",
        "analogue": "CHAMPION_MULTI_NWP_G12_ANALOGUE",
        "recipe_present": all(item["matches_expected"] for item in recipe_ids.values()),
        "source_manifest_identities_match": all(
            item["matches_expected"] for item in source_ids.values()
        ),
        "source_evidence": evidence,
        "recipe_identities": recipe_ids,
        "source_manifest_identities": source_ids,
        "blockers": blockers,
        "causal_rolling_origin_oof_estimable": False,
        "decision": "STOP",
        "permitted_next_action": (
            "Acquire and hash-lock an immutable as-issued 2022/2023 ECMWF+ICON+GFS "
            "bundle, or retire this analogue. Do not fit or score on 2024/2025."
        ),
    }


__all__ = [
    "EXPECTED_SOURCE_MANIFEST_SHA256",
    "RECIPE_IDENTITIES",
    "SOURCE_MANIFESTS",
    "assess_champion_reconstruction",
    "sha256_file",
]

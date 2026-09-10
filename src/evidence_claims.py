"""Evidence-only utilities for the BARAM2026 V2 audit.

This module deliberately has no model-selection API.  Public and user-reported
leaderboard values are evidence annotations only.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd


GROUP_COLUMNS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
SUBMISSION_COLUMNS = ("forecast_id", "forecast_kst_dtm", *GROUP_COLUMNS)
CLAIM_CLASSES = {
    "PUBLIC_SHA_LINKED",
    "PLATFORM_RECEIPT_VERIFIED",
    "LOCAL_REPRODUCED",
    "LOCAL_REPORTED_ONLY",
    "HYPOTHESIS",
}
PUBLIC_SELECTOR_ALLOWED = False


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def write_json_exclusive(path: Path, payload: Any) -> None:
    write_text_exclusive(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def write_csv_exclusive(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_parquet_exclusive(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    frame.to_parquet(path, index=False)


def read_compact_manifest(compact_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = compact_root / "MANIFEST_SHA256.csv"
    rows: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rel = Path(row["path"]).as_posix()
            if rel in rows:
                raise ValueError(f"duplicate compact manifest path: {rel}")
            rows[rel] = {
                "path": rel,
                "size_bytes": int(row["bytes"]),
                "sha256": row["sha256"].lower(),
            }
    return rows


def verify_compact_manifest(compact_root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for rel, declared in read_compact_manifest(compact_root).items():
        path = compact_root / rel
        row = dict(declared)
        row["exists"] = path.is_file()
        if path.is_file():
            actual = file_identity(path)
            row["actual_size_bytes"] = actual["size_bytes"]
            row["actual_sha256"] = actual["sha256"]
            row["status"] = (
                "FOUND_HASH_MATCH"
                if actual["size_bytes"] == declared["size_bytes"]
                and actual["sha256"] == declared["sha256"]
                else "FOUND_HASH_MISMATCH"
            )
        else:
            row.update(actual_size_bytes=None, actual_sha256=None, status="MISSING")
        result.append(row)
    return result


def validate_submission(path: Path, reference: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path)
    if tuple(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"unexpected submission columns in {path}: {tuple(frame.columns)}")
    if len(frame) != 8760:
        raise ValueError(f"submission row count is {len(frame)}, expected 8760: {path}")
    if frame["forecast_id"].duplicated().any():
        raise ValueError(f"duplicate forecast_id: {path}")
    timestamps = pd.to_datetime(frame["forecast_kst_dtm"], errors="raise")
    values = frame.loc[:, GROUP_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite prediction: {path}")
    if (values < 0).any():
        raise ValueError(f"negative prediction: {path}")
    if reference is not None:
        if not frame["forecast_id"].equals(reference["forecast_id"]):
            raise ValueError(f"forecast_id order mismatch: {path}")
        if not frame["forecast_kst_dtm"].equals(reference["forecast_kst_dtm"]):
            raise ValueError(f"timestamp order mismatch: {path}")
    return frame, {
        **file_identity(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "timestamp_min": timestamps.min().isoformat(),
        "timestamp_max": timestamps.max().isoformat(),
        "finite": True,
        "nonnegative": True,
    }


def verify_public_registry(compact_root: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    registry_path = compact_root / "PUBLIC_RESULTS.csv"
    registry = pd.read_csv(registry_path)
    if len(registry) != 9:
        raise ValueError(f"frozen Public registry must have 9 rows, got {len(registry)}")
    base_path = compact_root / "csv_scored" / "corrected_recent_v4.csv"
    base, _ = validate_submission(base_path)
    verified: list[dict[str, Any]] = []
    for record in registry.to_dict(orient="records"):
        csv_path = compact_root / "csv_scored" / str(record["csv_name"])
        _, validation = validate_submission(csv_path, base)
        arithmetic_error = abs(
            float(record["score"])
            - 0.5 * (float(record["one_minus_nmae"]) + float(record["ficr"]))
        )
        sha_match = validation["sha256"] == str(record["sha256"]).lower()
        if not sha_match or arithmetic_error > 1e-9:
            raise ValueError(f"invalid frozen Public row: {record['label']}")
        verified.append(
            {
                **record,
                "csv_path": csv_path.resolve().as_posix(),
                "csv_size_bytes": validation["size_bytes"],
                "arithmetic_error": arithmetic_error,
                "claim_class": "PUBLIC_SHA_LINKED",
                "platform_receipt_verified": False,
                "selector_eligible": False,
            }
        )
    return verified, base


def build_user_feedback_receipt(
    compact_root: Path,
    score: float,
    one_minus_nmae: float,
    ficr: float,
    submitted_kst: str | None = None,
) -> dict[str, Any]:
    csv_name = "multi_nwp_agreement_gate_g12_recent097_2025.csv"
    csv_path = compact_root / "csv_unscored_candidates" / csv_name
    base, _ = validate_submission(compact_root / "csv_scored" / "corrected_recent_v4.csv")
    _, validation = validate_submission(csv_path, base)
    arithmetic_error = abs(score - 0.5 * (one_minus_nmae + ficr))
    if arithmetic_error > 1e-9:
        raise ValueError("user-reported feedback score arithmetic is inconsistent")
    return {
        "schema_version": 1,
        "submission_id": "user_reported_agreement_gate_g12_20260810",
        "submitted_kst": submitted_kst,
        "csv_name": csv_name,
        "csv_sha256": validation["sha256"],
        "csv_size_bytes": validation["size_bytes"],
        "csv_path": csv_path.resolve().as_posix(),
        "score": score,
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
        "score_identity_error": arithmetic_error,
        "receipt_path": None,
        "receipt_sha256": None,
        "platform_filename": csv_name,
        "claim_class": "LOCAL_REPORTED_ONLY",
        "evidence_origin": "USER_REPORTED",
        "platform_receipt_verified": False,
        "selector_eligible": False,
        "public_registry_member": False,
        "notes": (
            "Metrics were supplied by the user without a structured platform receipt. "
            "They are preserved append-only for diagnosis and are forbidden as a selector, "
            "loss, threshold, ownership, or blend-weight input."
        ),
    }


def iter_path_sha_records(value: Any, source_manifest: Path) -> Iterator[dict[str, Any]]:
    """Yield nested manifest records that declare a path and SHA-256."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield {
                "declared_path": value["path"],
                "declared_sha256": value["sha256"].lower(),
                "declared_size_bytes": value.get("size_bytes", value.get("bytes")),
                "source_manifest": source_manifest.resolve().as_posix(),
            }
        for nested in value.values():
            yield from iter_path_sha_records(nested, source_manifest)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_path_sha_records(nested, source_manifest)


def manifest_registration_index(manifests: Iterable[Path]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for manifest in manifests:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for row in iter_path_sha_records(payload, manifest):
            declared = Path(row["declared_path"])
            if not declared.is_absolute():
                declared = (manifest.parent / declared).resolve()
            key = declared.resolve().as_posix().lower()
            index.setdefault(key, []).append(row)
    return index


def classify_identity(path: Path, registrations: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": path.resolve().as_posix(),
            "status": "MISSING",
            "size_bytes": None,
            "sha256": None,
            "registered_by": None,
        }
    actual = file_identity(path)
    declared = registrations.get(path.resolve().as_posix().lower(), [])
    status = "FOUND_UNREGISTERED"
    if declared:
        status = "FOUND_HASH_MISMATCH"
        for row in declared:
            size_ok = row["declared_size_bytes"] is None or int(row["declared_size_bytes"]) == actual["size_bytes"]
            if size_ok and row["declared_sha256"] == actual["sha256"]:
                status = "FOUND_HASH_MATCH"
                break
    return {
        **actual,
        "status": status,
        "registered_by": json.dumps(
            sorted({row["source_manifest"] for row in declared}), ensure_ascii=False
        ) if declared else None,
    }


def looks_like_submission(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
    except (OSError, UnicodeDecodeError, StopIteration):
        return False
    return tuple(header) == SUBMISSION_COLUMNS


def discover_submission_csvs(artifacts_root: Path) -> list[Path]:
    return sorted(
        (path for path in artifacts_root.rglob("*.csv") if looks_like_submission(path)),
        key=lambda path: path.as_posix().lower(),
    )

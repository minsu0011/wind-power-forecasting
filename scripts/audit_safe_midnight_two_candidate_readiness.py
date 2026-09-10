"""Read-only physical and lineage audit for the two-file midnight allowlist.

This script never fits, scores, submits, reads new leaderboard feedback, or writes
candidate CSVs.  Its only outputs are an audit JSON and its SHA-256 sidecar.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "artifacts/audits/safe_midnight_two_candidate_readiness_v1.json"
SIDECAR_PATH = AUDIT_PATH.with_suffix(".sha256")
SAMPLE_PATH = Path(r"data/local/open/sample_submission.csv")

EXPECTED_COLUMNS = [
    "forecast_id",
    "forecast_kst_dtm",
    "kpx_group_1",
    "kpx_group_2",
    "kpx_group_3",
]
TARGETS = EXPECTED_COLUMNS[2:]
CAPACITY_KWH = {
    "kpx_group_1": 21_600.0,
    "kpx_group_2": 21_600.0,
    "kpx_group_3": 21_000.0,
}
SIX_DECIMAL = re.compile(r"^(?:0|[1-9][0-9]*)\.[0-9]{6}$")

CANDIDATES = {
    "global_scale_097_plain": {
        "path": ROOT
        / "artifacts/postgate/public_adaptive_scale097_g2_delta_v2/"
        "corrected_recent_v4_scale_097_public_adaptive_2025.csv",
        "sha256": "e8f848b9a08c44b2552367f8e83ae4c9e132c8cf416ea8c437fd7a7e26314f58",
        "bytes": 624_176,
        "manifest": ROOT
        / "artifacts/postgate/public_adaptive_scale097_g2_delta_v2/"
        "manifest_frozen_postrun_v2.json",
        "manifest_sha256": "3ef6fabf059f3bd67e3b933e8eb633e5ebf3b422d7c123f1b27af461fcc2e153",
        "independent_audits": [
            {
                "path": ROOT / "artifacts/audits/public_adaptive_scale097_g2_delta_v2_raw_gpu.json",
                "sha256": "9e46b778af0131d576cf77a0ae3a94be3ca87e5549f2ecb0b472c6228def73e0",
                "required_status": "PASS",
            },
            {
                "path": ROOT
                / "artifacts/audits/public_adaptive_scale097_g2_delta_v2_postrun_independent.json",
                "sha256": "6f70538ce240afc46b1358bc379e635cadecff04c2151893417980258aad8e32",
                "required_status": "PASS",
            },
        ],
        "risk": {
            "strict": False,
            "selection_unsafe": True,
            "public_adaptive": True,
            "private_risk": "very_high",
            "qualification": (
                "The file/formula are frozen and audited, but it has no strict absolute forward gate. "
                "Any Public estimate is descriptive interpolation, not an identified score or Private claim."
            ),
        },
    },
    "direct_interval_g2_rescue_v5": {
        "path": ROOT
        / "artifacts/postgate/direct_interval_g2_posthoc_rescue_v5/"
        "direct_interval_g2_posthoc_rescue_v5_2025.csv",
        "sha256": "2596bf1048f7e1635c7c965096f896d5a4104f3e04b58f575e21f9e2314bfddf",
        "bytes": 624_444,
        "manifest": ROOT / "artifacts/postgate/direct_interval_g2_posthoc_rescue_v5/manifest.json",
        "manifest_sha256": "e02b966a0e02e06a8d12fc4ce23b129bde6c7f0f33e870231537b0b938d97f10",
        "independent_audits": [
            {
                "path": ROOT / "artifacts/audits/direct_interval_g2_posthoc_rescue_v5_independent.json",
                "sha256": "a73114afebdf4ec04b70f10d5a29fc32e697fd2f503b9c7702d8289d91aa03eb",
                "required_status": "PASS",
            }
        ],
        "risk": {
            "strict": False,
            "selection_unsafe": True,
            "posthoc_2023_group_selection": True,
            "posthoc_2024_g2_rescue": True,
            "private_risk": "high",
            "qualification": (
                "The fresh v5 AST closure and exact prediction/CSV replay passed, but the rescue is post-hoc; "
                "its smallest registered 2024 margins are tiny and Public/Private scores are unknown."
            ),
        },
    },
}

EVIDENCE = [
    {
        "path": ROOT / "artifacts/audits/midnight_hierarchical_public_probe_redteam_v1.json",
        "sha256": "706241a7b55f4681795b1c05544fdc37290b5460e90aa363f70006b41a7c5ec6",
        "purpose": "rule-risk exclusion of hierarchical group/time Public oracle probing",
    },
    {
        "path": ROOT / "artifacts/audits/slot5_structural2_existing_canonical_comparison_v1.json",
        "sha256": "7db7b489652494ff0250eb7cf487cc9869da4087e8862bf190cd6b40fcc53564",
        "purpose": "existing NO-GO evidence for corrected-v3, corrected hybrid, FICR, and low-CF candidates",
    },
    {
        "path": ROOT / "artifacts/audits/public_adaptive_scale097_ficr_g1_delta_v1_raw_gpu.json",
        "sha256": "d846ec762063f7ce4a8e2a9abc85d8804e4b89e5b6cf106ff71ba7927724be2d",
        "purpose": "independent integrity audit of rejected scale097-plus-FICR hybrid",
    },
    {
        "path": ROOT / "artifacts/audits/public_fixed_scale097_lowcf025_v1_local.json",
        "sha256": "c0a2318eeebe7cb3ee5dce3d5490182d149a7b9c1f14fa8e074738d8dbe0ebf3",
        "purpose": "NO-GO evidence for fixed scale097 low-CF candidate",
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def snapshot(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        rel(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(set(paths), key=lambda value: str(value).lower())
    }


def manifest_contains(manifest: Any, expected_sha: str, expected_bytes: int) -> bool:
    if isinstance(manifest, dict):
        if manifest.get("sha256") == expected_sha and manifest.get("size_bytes") == expected_bytes:
            return True
        return any(manifest_contains(value, expected_sha, expected_bytes) for value in manifest.values())
    if isinstance(manifest, list):
        return any(manifest_contains(value, expected_sha, expected_bytes) for value in manifest)
    return False


def audit_csv(path: Path, expected_sha: str, expected_bytes: int, sample: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = path.read_bytes()
    if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise AssertionError(f"physical identity changed: {path}")
    if not raw.startswith(b"\xef\xbb\xbf"):
        raise AssertionError(f"UTF-8 BOM missing: {path}")

    text = raw.decode("utf-8-sig")
    parsed = list(csv.reader(text.splitlines()))
    if parsed[0] != EXPECTED_COLUMNS or len(parsed) != 8_761:
        raise AssertionError(f"raw CSV schema/row count changed: {path}")
    if any(len(row) != len(EXPECTED_COLUMNS) for row in parsed):
        raise AssertionError(f"ragged CSV row: {path}")
    six_decimal = all(SIX_DECIMAL.fullmatch(value) is not None for row in parsed[1:] for value in row[2:])
    if not six_decimal:
        raise AssertionError(f"non-six-decimal prediction token: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"forecast_id": str, "forecast_kst_dtm": str})
    if frame.columns.tolist() != EXPECTED_COLUMNS or len(frame) != 8_760:
        raise AssertionError(f"parsed schema/row count changed: {path}")
    if not frame["forecast_id"].is_unique or not frame["forecast_kst_dtm"].is_unique:
        raise AssertionError(f"duplicate id/time: {path}")
    if not frame[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]]):
        raise AssertionError(f"id/time order differs from official sample: {path}")

    times = pd.to_datetime(frame["forecast_kst_dtm"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    expected_times = pd.date_range("2025-01-01 01:00:00", periods=8_760, freq="h")
    if not np.array_equal(times.to_numpy(), expected_times.to_numpy()):
        raise AssertionError(f"timestamp sequence changed: {path}")

    numeric = frame[TARGETS].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise AssertionError(f"non-finite values: {path}")
    lower_ok = bool((numeric >= 0.0).all())
    upper = np.asarray([1.02 * CAPACITY_KWH[target] for target in TARGETS], dtype=np.float64)
    upper_ok = bool((numeric <= upper[None, :] + 5e-7).all())
    if not lower_ok or not upper_ok:
        raise AssertionError(f"submission bounds changed: {path}")

    result = {
        "path": rel(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "utf8_bom": True,
        "line_ending": "CRLF" if b"\r\n" in raw else "LF",
        "schema_exact": True,
        "columns": EXPECTED_COLUMNS,
        "rows": len(frame),
        "six_decimal_prediction_tokens": True,
        "id_unique": True,
        "time_unique": True,
        "id_order_exact_sample": True,
        "time_order_exact_sample": True,
        "id_bounds": [frame["forecast_id"].iloc[0], frame["forecast_id"].iloc[-1]],
        "timestamp_bounds": [frame["forecast_kst_dtm"].iloc[0], frame["forecast_kst_dtm"].iloc[-1]],
        "hourly_operating_year_sequence_exact": True,
        "duplicate_full_rows": int(frame.duplicated().sum()),
        "finite_numeric": True,
        "lower_bound_nonnegative": lower_ok,
        "upper_bound_at_most_1p02_capacity": upper_ok,
        "value_bounds_kwh": {
            target: [float(frame[target].min()), float(frame[target].max())] for target in TARGETS
        },
    }
    return result, frame


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    implementation_path = Path(__file__).resolve()
    sample = pd.read_csv(
        SAMPLE_PATH,
        encoding="utf-8-sig",
        dtype={"forecast_id": str, "forecast_kst_dtm": str},
    )
    if sample.columns.tolist() != EXPECTED_COLUMNS or len(sample) != 8_760:
        raise AssertionError("official sample schema/rows changed")

    protected_paths = [SAMPLE_PATH]
    for spec in CANDIDATES.values():
        protected_paths.extend([spec["path"], spec["manifest"]])
        protected_paths.extend(item["path"] for item in spec["independent_audits"])
    protected_paths.extend(item["path"] for item in EVIDENCE)
    before = snapshot(protected_paths)

    candidates: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    upstream: dict[str, Any] = {}
    for name, spec in CANDIDATES.items():
        physical, frame = audit_csv(spec["path"], spec["sha256"], spec["bytes"], sample)
        frames[name] = frame

        manifest_hash = sha256(spec["manifest"])
        if manifest_hash != spec["manifest_sha256"]:
            raise AssertionError(f"manifest changed: {spec['manifest']}")
        manifest = load_json(spec["manifest"])
        manifest_bound = manifest_contains(manifest, spec["sha256"], spec["bytes"])
        if not manifest_bound:
            raise AssertionError(f"manifest does not bind CSV: {spec['manifest']}")

        audit_records = []
        for item in spec["independent_audits"]:
            actual_hash = sha256(item["path"])
            if actual_hash != item["sha256"]:
                raise AssertionError(f"independent audit changed: {item['path']}")
            audit = load_json(item["path"])
            if audit.get("status") != item["required_status"]:
                raise AssertionError(f"independent audit is not PASS: {item['path']}")
            audit_records.append(
                {
                    "path": rel(item["path"]),
                    "bytes": item["path"].stat().st_size,
                    "sha256": actual_hash,
                    "status": audit["status"],
                }
            )

        candidates[name] = {
            "physical": physical,
            "authoritative_manifest": {
                "path": rel(spec["manifest"]),
                "bytes": spec["manifest"].stat().st_size,
                "sha256": manifest_hash,
                "csv_record_exact": manifest_bound,
            },
            "independent_audits": audit_records,
            "risk": spec["risk"],
            "ready_for_manual_submission": True,
            "leaderboard_score_claim": False,
            "private_champion_claim": False,
        }

    a = frames["global_scale_097_plain"]
    b = frames["direct_interval_g2_rescue_v5"]
    av = a[TARGETS].to_numpy(dtype=np.float64)
    bv = b[TARGETS].to_numpy(dtype=np.float64)
    differences = av != bv
    pairwise = {
        "byte_identical": CANDIDATES["global_scale_097_plain"]["sha256"]
        == CANDIDATES["direct_interval_g2_rescue_v5"]["sha256"],
        "sha256_distinct": CANDIDATES["global_scale_097_plain"]["sha256"]
        != CANDIDATES["direct_interval_g2_rescue_v5"]["sha256"],
        "numeric_duplicate": bool(np.array_equal(av, bv)),
        "different_numeric_cells": int(differences.sum()),
        "different_rows": int(differences.any(axis=1).sum()),
        "different_cells_by_group": {
            target: int(differences[:, position].sum()) for position, target in enumerate(TARGETS)
        },
        "max_absolute_difference_kwh_by_group": {
            target: float(np.max(np.abs(av[:, position] - bv[:, position])))
            for position, target in enumerate(TARGETS)
        },
    }
    if pairwise["numeric_duplicate"] or not pairwise["sha256_distinct"]:
        raise AssertionError("the two allowlisted files are duplicates")

    evidence_records = []
    for item in EVIDENCE:
        actual = sha256(item["path"])
        if actual != item["sha256"]:
            raise AssertionError(f"risk evidence changed: {item['path']}")
        evidence_records.append(
            {
                "path": rel(item["path"]),
                "bytes": item["path"].stat().st_size,
                "sha256": actual,
                "purpose": item["purpose"],
            }
        )

    after = snapshot(protected_paths)
    if before != after:
        raise AssertionError("a protected candidate/upstream file changed during audit")

    report = {
        "schema_version": 1,
        "audit_id": "safe_midnight_two_candidate_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "READY_TWO_FILE_ALLOWLIST_WITH_FIXED_ORDER_AND_RULE_RISK_EXCLUSIONS",
        "scope": {
            "candidate_files_audited": 2,
            "candidate_CSVs_created_or_modified": 0,
            "new_feedback_reads_or_calls": 0,
            "submission_or_upload_calls": 0,
            "fit_calls": 0,
            "target_or_candidate_score_calls": 0,
            "canonical_files_modified": 0,
            "only_outputs": [rel(AUDIT_PATH), rel(SIDECAR_PATH)],
        },
        "audit_implementation": {
            "path": rel(implementation_path),
            "bytes": implementation_path.stat().st_size,
            "sha256": sha256(implementation_path),
        },
        "official_sample": {
            "path": rel(SAMPLE_PATH),
            "bytes": SAMPLE_PATH.stat().st_size,
            "sha256": sha256(SAMPLE_PATH),
            "rows": len(sample),
            "schema_exact": sample.columns.tolist() == EXPECTED_COLUMNS,
        },
        "candidates": candidates,
        "pairwise_nonduplication": pairwise,
        "fixed_manual_sequence": [
            {
                "step": 1,
                "action": "manually submit only global_scale_097_plain",
                "required_sha256": CANDIDATES["global_scale_097_plain"]["sha256"],
            },
            {
                "step": 2,
                "action": (
                    "wait for and record the actual displayed total_score, one_minus_nmae, and FICR triplet "
                    "bound to the exact step-1 SHA; do not infer group/time/quarter labels and do not generate a hybrid"
                ),
            },
            {
                "step": 3,
                "action": "manually submit only direct_interval_g2_rescue_v5",
                "required_sha256": CANDIDATES["direct_interval_g2_rescue_v5"]["sha256"],
            },
        ],
        "fixed_order_is_not_adaptive_oracle_search": True,
        "rule_risk_exclusion": {
            "hierarchical_group_time_pack": "EXCLUDED_STOP",
            "reason": (
                "The red-team found substantial risk that deliberate group-to-half-to-quarter Public inversion "
                "constitutes prohibited answer-equivalent evaluation-data probing."
            ),
            "no_group_only_probe": True,
            "no_half_or_quarter_probe": True,
            "no_exact_group_or_time_inference": True,
            "no_feedback-generated_hybrid": True,
        },
        "existing_no_go_exclusions": {
            "corrected_v3": "NO_GO; existing read-only comparison records severe 2023-to-2024 reversal and uniformly negative 2024 evidence.",
            "corrected_hybrid": "NO_GO; broad hedge with only 1/7 positive 2024 slices in the existing comparison.",
            "FICR_candidates_and_hybrids": "NO_GO for this allowlist; existing evidence records slice downside, adverse Public evidence, or diagnostic_GO=false.",
            "low_CF_candidates": "NO_GO; fixed low-CF candidates failed Stage1 or the G1/G3 rescue failed both locked 2024 baselines.",
            "selection_contract": "None of these excluded files may be substituted into the two-file sequence without a new explicit audit.",
        },
        "risk_evidence": evidence_records,
        "residual_PMF_followup": {
            "current_allowlist_effect": "none",
            "conditional_plan": (
                "If and only if the separate residual-PMF experiment passes its frozen strict protocol, "
                "issue a separate append-only readiness addendum before considering it."
            ),
            "no_presumed_pass_or_slot": True,
        },
        "nonmutation": {
            "protected_snapshot_before_equals_after": before == after,
            "protected_file_count": len(before),
            "snapshot": after,
        },
        "overall_risk": {
            "strict_final_isolation": False,
            "selection_unsafe": True,
            "Public40_Private60_overfit_risk": "high_to_very_high",
            "leaderboard_score_claim": False,
            "private_champion_claim": False,
            "manual_operator_must_recheck_SHA_before_each_upload": True,
        },
    }

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha256(AUDIT_PATH)
    SIDECAR_PATH.write_text(f"{digest}  {AUDIT_PATH.name}\n", encoding="ascii")
    print(json.dumps({"audit": rel(AUDIT_PATH), "bytes": AUDIT_PATH.stat().st_size, "sha256": digest}))


if __name__ == "__main__":
    main()

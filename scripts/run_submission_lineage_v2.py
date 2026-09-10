"""Build the BARAM2026 Phase-A submission lineage graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_claims import (  # noqa: E402
    GROUP_COLUMNS,
    discover_submission_csvs,
    file_identity,
    verify_public_registry,
    write_json_exclusive,
    write_parquet_exclusive,
    write_text_exclusive,
)
from src.submission_lineage import (  # noqa: E402
    affine_metrics,
    changed_slices,
    effective_rank,
    increment_similarity,
    load_candidate,
    mechanism_family,
    preferred_parent_name,
    verify_required_lineage_facts,
    write_graphml,
)


def _resolve(workspace: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _priority(path: Path, compact: Path) -> tuple[int, str]:
    text = path.resolve().as_posix().lower()
    return (0 if text.startswith(compact.resolve().as_posix().lower()) else 1, text)


def _root_family(family: str) -> str:
    if family.startswith("multi_nwp"):
        return "multi_nwp_external_axis"
    if family in {"probability_calibration", "bayes_decision"}:
        return "probabilistic_ficr_axis"
    if family == "global_or_group_scalar_calibration":
        return "scalar_calibration_axis"
    return family


def _safe_number(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("phase") != "A":
        raise ValueError("this runner is Phase A only")
    compact = _resolve(ROOT, config["compact_root"])
    output = _resolve(ROOT, config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    public_rows, canonical_base = verify_public_registry(compact)
    public_by_sha = {row["sha256"]: row for row in public_rows}
    feedback = config["user_feedback"]
    feedback_sha = "8d415219350449d4d1b4b4c89d7e8d7269b1fd4636ba5589cf5ea5f1cb251c5e"

    paths = discover_submission_csvs(ROOT / "artifacts")
    candidates = []
    skipped: list[dict[str, str]] = []
    for path in paths:
        try:
            candidates.append(load_candidate(path, canonical_base))
        except Exception as exc:
            skipped.append({"path": path.resolve().as_posix(), "error": f"{type(exc).__name__}: {exc}"})
    if not candidates:
        raise ValueError("no valid submission CSV found")

    candidates.sort(key=lambda item: _priority(item.path, compact))
    node_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        public = public_by_sha.get(candidate.identity["sha256"])
        is_feedback = candidate.identity["sha256"] == feedback_sha
        family = mechanism_family(candidate.path)
        node_rows.append(
            {
                "node_id": candidate.node_id,
                "csv_name": candidate.path.name,
                "csv_path": candidate.path.as_posix(),
                "csv_size_bytes": candidate.identity["size_bytes"],
                "csv_sha256": candidate.identity["sha256"],
                "prediction_hash_full": candidate.prediction_hash_full,
                "prediction_hash_g1": candidate.prediction_hash_group[GROUP_COLUMNS[0]],
                "prediction_hash_g2": candidate.prediction_hash_group[GROUP_COLUMNS[1]],
                "prediction_hash_g3": candidate.prediction_hash_group[GROUP_COLUMNS[2]],
                "mechanism_family": family,
                "independent_axis": _root_family(family),
                "claim_class": "PUBLIC_SHA_LINKED" if public else ("LOCAL_REPORTED_ONLY" if is_feedback else "LOCAL_REPRODUCED"),
                "score": float(public["score"]) if public else (float(feedback["score"]) if is_feedback else None),
                "one_minus_nmae": float(public["one_minus_nmae"]) if public else (float(feedback["one_minus_nmae"]) if is_feedback else None),
                "ficr": float(public["ficr"]) if public else (float(feedback["ficr"]) if is_feedback else None),
                "platform_receipt_verified": False,
                "selector_eligible": False,
                "user_reported_feedback": bool(is_feedback),
            }
        )
    nodes = pd.DataFrame(node_rows)
    by_id = {candidate.node_id: candidate for candidate in candidates}

    # Representatives are deterministic and prefer the frozen compact package.
    rep_by_prediction: dict[str, Any] = {}
    rep_by_sha: dict[str, Any] = {}
    for candidate in candidates:
        rep_by_prediction.setdefault(candidate.prediction_hash_full, candidate)
        rep_by_sha.setdefault(candidate.identity["sha256"], candidate)
    name_lookup: dict[str, Any] = {}
    for candidate in candidates:
        name_lookup.setdefault(candidate.path.name.lower(), candidate)

    edge_rows: list[dict[str, Any]] = []
    for sha, members in nodes.groupby("csv_sha256", sort=True):
        parent_id = rep_by_sha[sha].node_id
        for child_id in members["node_id"]:
            if child_id != parent_id:
                edge_rows.append(
                    {
                        "parent_node_id": parent_id,
                        "child_node_id": child_id,
                        "relation_type": "EXACT_SHA_DUPLICATE",
                        "scope": "full",
                        "decision": "DUPLICATE_MECHANISM_REJECTED",
                        "exact_prediction_identity": True,
                    }
                )
    for prediction_hash, members in nodes.groupby("prediction_hash_full", sort=True):
        parent_id = rep_by_prediction[prediction_hash].node_id
        for child_id in members["node_id"]:
            if child_id != parent_id and not any(
                row["parent_node_id"] == parent_id and row["child_node_id"] == child_id
                for row in edge_rows
            ):
                edge_rows.append(
                    {
                        "parent_node_id": parent_id,
                        "child_node_id": child_id,
                        "relation_type": "EXACT_PREDICTION_DUPLICATE",
                        "scope": "full",
                        "decision": "DUPLICATE_MECHANISM_REJECTED",
                        "exact_prediction_identity": True,
                    }
                )

    representatives = list(rep_by_prediction.values())
    base = name_lookup.get("corrected_recent_v4.csv") or name_lookup.get("recent_v4.csv")
    scale097 = name_lookup.get("corrected_recent_v4_scale_097_public_adaptive_2025.csv")
    if base is None or scale097 is None:
        raise ValueError("canonical recent-v4/scale097 parents missing")

    for child in representatives:
        parent_name = preferred_parent_name(child.path.name)
        if parent_name is None:
            continue
        parent = name_lookup.get(parent_name.lower())
        if parent is None or parent.prediction_hash_full == child.prediction_hash_full:
            continue
        family = mechanism_family(child.path)
        group_json, month_json, lead_json = changed_slices(parent.frame, child.frame)
        for scope in ("full", *GROUP_COLUMNS):
            if scope == "full":
                p_values = parent.frame.loc[:, GROUP_COLUMNS].to_numpy(dtype=np.float64)
                c_values = child.frame.loc[:, GROUP_COLUMNS].to_numpy(dtype=np.float64)
            else:
                p_values = parent.frame[scope].to_numpy(dtype=np.float64)
                c_values = child.frame[scope].to_numpy(dtype=np.float64)
            metrics = {key: _safe_number(value) for key, value in affine_metrics(p_values, c_values).items()}
            no_new_source = family in {
                "global_or_group_scalar_calibration", "multi_nwp_agreement_gate",
                "multi_nwp_disjoint_ownership", "hybrid_blend", "temporal_smoothing",
            }
            duplicate = bool(
                no_new_source
                and (
                    (metrics["pearson"] is not None and metrics["pearson"] > 0.9995)
                    or (metrics["changed_fraction"] is not None and metrics["changed_fraction"] < 0.02)
                )
            )
            edge_rows.append(
                {
                    "parent_node_id": parent.node_id,
                    "child_node_id": child.node_id,
                    "relation_type": "PLAUSIBLE_PARENT_AFFINE_DIAGNOSTIC",
                    "scope": scope,
                    "decision": "DUPLICATE_MECHANISM_REJECTED" if duplicate else "DISTINCT_OR_UNRESOLVED",
                    "exact_prediction_identity": bool(np.array_equal(p_values, c_values)),
                    "changed_by_group_json": group_json if scope == "full" else None,
                    "changed_by_month_json": month_json if scope == "full" else None,
                    "changed_by_lead_bin_json": lead_json if scope == "full" else None,
                    **metrics,
                }
            )

    # Compare increments within a conceptual axis against the same recent-v4 base.
    families: dict[str, list[Any]] = defaultdict(list)
    for candidate in representatives:
        families[_root_family(mechanism_family(candidate.path))].append(candidate)
    for family, members in families.items():
        if len(members) < 2:
            continue
        for left_index in range(len(members)):
            for right_index in range(left_index + 1, len(members)):
                left, right = members[left_index], members[right_index]
                similarity = increment_similarity(
                    base.frame.loc[:, GROUP_COLUMNS].to_numpy(),
                    left.frame.loc[:, GROUP_COLUMNS].to_numpy(),
                    right.frame.loc[:, GROUP_COLUMNS].to_numpy(),
                )
                corr = similarity["increment_corr"]
                strong = bool(similarity["increment_exact"] or (corr is not None and abs(corr) > 0.95))
                if strong:
                    edge_rows.append(
                        {
                            "parent_node_id": left.node_id,
                            "child_node_id": right.node_id,
                            "relation_type": "CONCEPTUAL_INCREMENT_DUPLICATE",
                            "scope": "full",
                            "decision": "DUPLICATE_MECHANISM_REJECTED",
                            "exact_prediction_identity": left.prediction_hash_full == right.prediction_hash_full,
                            "independent_axis": family,
                            **similarity,
                        }
                    )

    edges = pd.DataFrame(edge_rows)
    for column in (
        "affine_a", "affine_b", "affine_max_abs_residual", "pearson", "spearman",
        "rmse", "mae", "max_abs_delta", "changed_rows", "changed_fraction",
        "increment_corr", "increment_exact", "nonzero_overlap_count",
        "nonzero_overlap_jaccard", "sign_agreement_on_overlap", "independent_axis",
        "changed_by_group_json", "changed_by_month_json", "changed_by_lead_bin_json",
    ):
        if column not in edges:
            edges[column] = None

    facts = verify_required_lineage_facts(compact)
    if not facts["all_required_facts_pass"]:
        raise ValueError(f"required lineage facts failed: {facts}")
    base_values = base.frame.loc[:, GROUP_COLUMNS].to_numpy(dtype=np.float64)
    rank = effective_rank(
        candidate.frame.loc[:, GROUP_COLUMNS].to_numpy(dtype=np.float64) - base_values
        for candidate in representatives
        if candidate.prediction_hash_full != base.prediction_hash_full
    )

    write_parquet_exclusive(output / "SUBMISSION_LINEAGE_NODES.parquet", nodes)
    write_parquet_exclusive(output / "SUBMISSION_LINEAGE_EDGES.parquet", edges)
    write_graphml(output / "SUBMISSION_LINEAGE_GRAPH.graphml", nodes, edges)
    write_json_exclusive(output / "SUBMISSION_LINEAGE_REQUIRED_FACTS.json", facts)

    unique_sha = int(nodes["csv_sha256"].nunique())
    unique_prediction = int(nodes["prediction_hash_full"].nunique())
    independent_axes = sorted(set(nodes["independent_axis"]))
    duplicate_decisions = int((edges["decision"] == "DUPLICATE_MECHANISM_REJECTED").sum())
    report = [
        "# Submission Lineage Report — Phase A",
        "",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Census",
        "",
        f"- Valid submission-shaped CSV paths: `{len(nodes)}`",
        f"- Skipped invalid submission-shaped CSVs: `{len(skipped)}`",
        f"- Unique file SHA-256 identities: `{unique_sha}`",
        f"- Unique full prediction tensors: `{unique_prediction}`",
        f"- Independent conceptual axes/hypotheses: `{len(independent_axes)}`",
        f"- Duplicate-mechanism edge decisions: `{duplicate_decisions}`",
        f"- Frozen Public rows linked: `{int((nodes['claim_class'] == 'PUBLIC_SHA_LINKED').sum())}` path copies (`9` unique SHA facts)",
        f"- User-reported agreement paths: `{int(nodes['user_reported_feedback'].sum())}`; selector eligible: `false`",
        "",
        "## Independent axes",
        "",
        *[f"- `{axis}`" for axis in independent_axes],
        "",
        "## Required exact facts",
        "",
        f"- scale .92/.95/.97/.98 quantized scalar identity: `{facts['scale_quantized_exact']}`",
        f"- probability/Bayes G1/G2 exact identity: `{facts['probability_bayes_exact_identity']}`",
        f"- direct interval changed rows: `{facts['direct_interval_changed_rows']}`",
        f"- joint/agreement/disjoint G3 identity to .97: `{facts['multi_nwp_g3_exact_identity_to_scale097']}`",
        f"- disjoint G1 increment identity to joint: `{facts['disjoint_g1_increment_exact_identity_to_joint']}`",
        "",
        "## Increment-space rank",
        "",
        f"- `{rank}`",
        "",
        "## Evidence boundary",
        "",
        "All leaderboard values are annotations. `selector_eligible=false` for every node. The new agreement metrics are `LOCAL_REPORTED_ONLY`, are absent from the frozen 9-row Public registry, and were not used to choose a parent, threshold, mechanism, ownership, blend, or hypothesis count. Parent relations are filename/config lineage diagnostics, not performance selections.",
        "",
    ]
    write_text_exclusive(output / "SUBMISSION_LINEAGE_REPORT.md", "\n".join(report))

    summary = {
        "schema_version": 1,
        "phase": "A_SUBMISSION_LINEAGE",
        "status": "PASS",
        "valid_csv_paths": len(nodes),
        "skipped_csv_paths": skipped,
        "unique_csv_sha256": unique_sha,
        "unique_prediction_tensors": unique_prediction,
        "independent_hypothesis_count": len(independent_axes),
        "independent_hypotheses": independent_axes,
        "duplicate_mechanism_edge_count": duplicate_decisions,
        "required_facts": facts,
        "increment_effective_rank": rank,
        "public_selector_used": False,
        "user_feedback_selector_used": False,
    }
    write_json_exclusive(output / "SUBMISSION_LINEAGE_SUMMARY.json", summary)

    output_files = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name.startswith("SUBMISSION_LINEAGE")
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "phase_a_submission_lineage",
        "code": file_identity(Path(__file__)),
        "config": file_identity(config_path),
        "outputs": [file_identity(path) for path in output_files],
        "public_selector_used": False,
        "user_feedback_selector_used": False,
    }
    write_json_exclusive(output / "SUBMISSION_LINEAGE_MANIFEST.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


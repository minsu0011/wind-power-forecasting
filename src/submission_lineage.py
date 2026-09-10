"""Target-free submission lineage diagnostics for BARAM2026 Phase A."""

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.evidence_claims import GROUP_COLUMNS, file_identity, validate_submission


@dataclass(frozen=True)
class Candidate:
    node_id: str
    path: Path
    frame: pd.DataFrame
    identity: dict[str, Any]
    prediction_hash_full: str
    prediction_hash_group: dict[str, str]


def _array_hash(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def load_candidate(path: Path, reference: pd.DataFrame | None = None) -> Candidate:
    frame, identity = validate_submission(path, reference)
    values = frame.loc[:, GROUP_COLUMNS].to_numpy(dtype=np.float64)
    full_hash = _array_hash(values)
    group_hash = {column: _array_hash(frame[column].to_numpy(dtype=np.float64)) for column in GROUP_COLUMNS}
    node_id = f"csv_{identity['sha256'][:16]}_{hashlib.sha256(path.resolve().as_posix().encode()).hexdigest()[:10]}"
    return Candidate(node_id, path.resolve(), frame, identity, full_hash, group_hash)


def quantized_scalar_identity(parent: np.ndarray, child: np.ndarray, scale: float, decimals: int = 6) -> bool:
    """Exact identity under the information interval of decimal CSV serialization.

    A six-decimal parent does not retain its pre-serialization value.  This test
    proves that every child cell is feasible under scalar multiplication plus
    one six-decimal serialization at each endpoint; it is stricter than a free
    allclose tolerance and binds the declared scale.
    """
    unit = 10.0 ** (-decimals)
    bound = 0.5 * unit * (1.0 + abs(scale)) + 8 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(child))
    return bool(np.all(np.abs(child - scale * parent) <= bound))


def affine_metrics(parent: np.ndarray, child: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(parent, dtype=np.float64).reshape(-1)
    y = np.asarray(child, dtype=np.float64).reshape(-1)
    delta = y - x
    changed = np.abs(delta) > 0.0
    if np.var(x) == 0:
        slope, intercept = math.nan, math.nan
    else:
        slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
        intercept = float(y.mean() - slope * x.mean())
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else math.nan
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(xr, yr)[0, 1]) if np.std(xr) and np.std(yr) else math.nan
    residual = y - (slope * x + intercept) if np.isfinite(slope) else np.full_like(y, np.nan)
    return {
        "affine_a": slope,
        "affine_b": intercept,
        "affine_max_abs_residual": float(np.nanmax(np.abs(residual))) if np.isfinite(residual).any() else None,
        "pearson": pearson,
        "spearman": spearman,
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mae": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "changed_rows": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
    }


def changed_slices(reference: pd.DataFrame, candidate: pd.DataFrame) -> tuple[str, str, str]:
    timestamps = pd.to_datetime(reference["forecast_kst_dtm"])
    month: dict[str, int] = {}
    group: dict[str, int] = {}
    lead: dict[str, int] = {}
    # Operating day is 01:00 through next 00:00; this gives deterministic D-1 lead slots.
    lead_number = np.where(timestamps.dt.hour.to_numpy() == 0, 24, timestamps.dt.hour.to_numpy())
    lead_labels = pd.cut(lead_number, [0, 6, 12, 18, 24], labels=["L01_06", "L07_12", "L13_18", "L19_24"])
    any_changed = np.zeros(len(reference), dtype=bool)
    for column in GROUP_COLUMNS:
        mask = reference[column].to_numpy() != candidate[column].to_numpy()
        group[column] = int(mask.sum())
        any_changed |= mask
    month_series = timestamps.dt.strftime("%Y-%m")
    for key in sorted(month_series.unique()):
        month[key] = int(any_changed[month_series.to_numpy() == key].sum())
    for key in ("L01_06", "L07_12", "L13_18", "L19_24"):
        lead[key] = int(any_changed[np.asarray(lead_labels == key)].sum())
    return (
        json.dumps(group, sort_keys=True),
        json.dumps(month, sort_keys=True),
        json.dumps(lead, sort_keys=True),
    )


def increment_similarity(base: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, float | int | None]:
    a = np.asarray(left, dtype=np.float64).reshape(-1) - np.asarray(base, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1) - np.asarray(base, dtype=np.float64).reshape(-1)
    a_nz, b_nz = a != 0, b != 0
    union = a_nz | b_nz
    both = a_nz & b_nz
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None
    return {
        "increment_corr": corr,
        "increment_exact": bool(np.array_equal(a, b)),
        "nonzero_overlap_count": int(both.sum()),
        "nonzero_overlap_jaccard": float(both.sum() / union.sum()) if union.any() else 1.0,
        "sign_agreement_on_overlap": float((np.sign(a[both]) == np.sign(b[both])).mean()) if both.any() else 1.0,
    }


def mechanism_family(path: Path) -> str:
    name = path.name.lower()
    if "agreement" in name:
        return "multi_nwp_agreement_gate"
    if "disjoint" in name:
        return "multi_nwp_disjoint_ownership"
    if "multi_nwp_joint" in name:
        return "multi_nwp_joint"
    if "multi_nwp_consensus" in name:
        return "multi_nwp_consensus"
    if re.search(r"scale_?0?(92|95|96|97|98)", name):
        return "global_or_group_scalar_calibration"
    if "prob" in name:
        return "probability_calibration"
    if "bayes" in name:
        return "bayes_decision"
    if "direct_interval" in name:
        return "direct_interval_g2"
    if "recency" in name:
        return "recency_weighting"
    if "turbine" in name or "scada" in name:
        return "turbine_scada"
    if "smoothed" in name:
        return "temporal_smoothing"
    if "hybrid" in name:
        return "hybrid_blend"
    if "corrected_v3" in name or "v3_locked" in name or "v3_existing" in name:
        return "corrected_v3_baseline"
    if name in {"corrected_recent_v4.csv", "recent_v4.csv"}:
        return "corrected_recent_v4_baseline"
    return "other_candidate"


def preferred_parent_name(candidate_name: str) -> str | None:
    name = candidate_name.lower()
    if name in {"corrected_recent_v4.csv", "recent_v4.csv"}:
        return None
    if any(token in name for token in ("multi_nwp", "recent097", "scale_097_plus")):
        return "corrected_recent_v4_scale_097_public_adaptive_2025.csv"
    if "v3" in name and "recent" not in name:
        return None
    return "corrected_recent_v4.csv"


def verify_required_lineage_facts(compact_root: Path) -> dict[str, Any]:
    scored = compact_root / "csv_scored"
    unscored = compact_root / "csv_unscored_candidates"
    base = pd.read_csv(scored / "corrected_recent_v4.csv")
    values = base.loc[:, GROUP_COLUMNS].to_numpy(dtype=np.float64)
    scalar: dict[str, bool] = {}
    for label, scale, filename in (
        ("092", 0.92, "corrected_recent_v4_scale_092_2025.csv"),
        ("095", 0.95, "corrected_recent_v4_scale_095_2025.csv"),
        ("097", 0.97, "corrected_recent_v4_scale_097_public_adaptive_2025.csv"),
        ("098", 0.98, "corrected_recent_v4_scale_098_bracket_2025.csv"),
    ):
        child = pd.read_csv(scored / filename).loc[:, GROUP_COLUMNS].to_numpy(dtype=np.float64)
        scalar[label] = quantized_scalar_identity(values, child, scale)
    probability = pd.read_csv(scored / "prob_stable_g1g3_2025.csv")
    bayes = pd.read_csv(scored / "ficr_bayes_decision_2025.csv")
    prob_bayes = {
        column: bool(np.array_equal(probability[column].to_numpy(), bayes[column].to_numpy()))
        for column in GROUP_COLUMNS[:2]
    }
    direct = pd.read_csv(scored / "direct_interval_g2_posthoc_rescue_v5_2025.csv")
    direct_changed = {
        column: int(np.count_nonzero(direct[column].to_numpy() != base[column].to_numpy()))
        for column in GROUP_COLUMNS
    }
    scale097 = pd.read_csv(scored / "corrected_recent_v4_scale_097_public_adaptive_2025.csv")
    joint = pd.read_csv(scored / "multi_nwp_joint_g12_posthoc_rescue_recent097_2025.csv")
    agreement = pd.read_csv(unscored / "multi_nwp_agreement_gate_g12_recent097_2025.csv")
    disjoint = pd.read_csv(unscored / "multi_nwp_disjoint_g1joint_g2consensus_2025.csv")
    g3_identity = {
        "joint": bool(np.array_equal(joint[GROUP_COLUMNS[2]].to_numpy(), scale097[GROUP_COLUMNS[2]].to_numpy())),
        "agreement": bool(np.array_equal(agreement[GROUP_COLUMNS[2]].to_numpy(), scale097[GROUP_COLUMNS[2]].to_numpy())),
        "disjoint": bool(np.array_equal(disjoint[GROUP_COLUMNS[2]].to_numpy(), scale097[GROUP_COLUMNS[2]].to_numpy())),
    }
    joint_inc = joint[GROUP_COLUMNS[0]].to_numpy() - scale097[GROUP_COLUMNS[0]].to_numpy()
    disjoint_inc = disjoint[GROUP_COLUMNS[0]].to_numpy() - scale097[GROUP_COLUMNS[0]].to_numpy()
    facts = {
        "scale_quantized_exact": scalar,
        "probability_bayes_exact_identity": prob_bayes,
        "direct_interval_changed_rows": direct_changed,
        "multi_nwp_g3_exact_identity_to_scale097": g3_identity,
        "disjoint_g1_increment_exact_identity_to_joint": bool(np.array_equal(joint_inc, disjoint_inc)),
    }
    passed = (
        all(scalar.values())
        and all(prob_bayes.values())
        and direct_changed == {"kpx_group_1": 0, "kpx_group_2": 143, "kpx_group_3": 0}
        and all(g3_identity.values())
        and facts["disjoint_g1_increment_exact_identity_to_joint"]
    )
    facts["all_required_facts_pass"] = bool(passed)
    return facts


def effective_rank(increments: Iterable[np.ndarray]) -> dict[str, float | int]:
    rows = [np.asarray(row, dtype=np.float64).reshape(-1) for row in increments]
    if not rows:
        return {"matrix_rows": 0, "numerical_rank": 0, "pca95_components": 0, "pca99_components": 0, "entropy_effective_rank": 0.0}
    matrix = np.stack(rows)
    gram = matrix @ matrix.T
    eig = np.maximum(np.linalg.eigvalsh(gram), 0.0)[::-1]
    total = float(eig.sum())
    if total == 0:
        return {"matrix_rows": len(rows), "numerical_rank": 0, "pca95_components": 0, "pca99_components": 0, "entropy_effective_rank": 0.0}
    share = eig / total
    cumulative = np.cumsum(share)
    positive = share[share > 0]
    return {
        "matrix_rows": len(rows),
        "numerical_rank": int(np.sum(eig > eig[0] * 1e-12)),
        "pca95_components": int(np.searchsorted(cumulative, 0.95) + 1),
        "pca99_components": int(np.searchsorted(cumulative, 0.99) + 1),
        "entropy_effective_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
    }


def write_graphml(path: Path, nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    namespace = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace("", namespace)
    root = ET.Element(f"{{{namespace}}}graphml")
    for key_id, name in (("label", "label"), ("sha256", "sha256"), ("relation", "relation")):
        ET.SubElement(root, f"{{{namespace}}}key", id=key_id, **{"for": "all", "attr.name": name, "attr.type": "string"})
    graph = ET.SubElement(root, f"{{{namespace}}}graph", id="submission_lineage", edgedefault="directed")
    for row in nodes.to_dict(orient="records"):
        node = ET.SubElement(graph, f"{{{namespace}}}node", id=str(row["node_id"]))
        for key, value in (("label", row["csv_name"]), ("sha256", row["csv_sha256"])):
            ET.SubElement(node, f"{{{namespace}}}data", key=key).text = str(value)
    for index, row in enumerate(edges.to_dict(orient="records")):
        edge = ET.SubElement(
            graph, f"{{{namespace}}}edge", id=f"e{index:05d}",
            source=str(row["parent_node_id"]), target=str(row["child_node_id"]),
        )
        ET.SubElement(edge, f"{{{namespace}}}data", key="relation").text = str(row["relation_type"])
    tree = ET.ElementTree(root)
    with path.open("xb") as handle:
        tree.write(handle, encoding="utf-8", xml_declaration=True)


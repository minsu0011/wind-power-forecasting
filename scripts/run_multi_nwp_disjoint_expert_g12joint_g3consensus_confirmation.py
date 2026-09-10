"""Evaluate the frozen disjoint G12-joint/G3-consensus 2024-H2 candidate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_ficr_bayes_decision_strict as strict  # noqa: E402
from src.manifest import describe_file, sha256_file, utc_now  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402


CONFIG = PROJECT_DIR / "configs/multi_nwp_disjoint_expert_g12joint_g3consensus_preregister_v2.json"
CONFIG_SHA256 = "bc13e6cc6b13ec6fd850b9e17e08bd3288097fcb5b967060ae05c10e6878ddb8"
JOINT = PROJECT_DIR / "artifacts/postgate/multi_nwp_joint_disagreement_2024_forward_v1/predictions/joint_paired_increment_cf.parquet"
JOINT_SHA256 = "14419ac0a939af154f334b2dc543fef0085e29be863a96432e5fdfd3378fb191"
CONSENSUS = PROJECT_DIR / "artifacts/postgate/multi_nwp_consensus_g23_rescue_v1/validation/consensus_increment_cf_2024_h2.parquet"
CONSENSUS_SHA256 = "932a5030dd87741fb21bef0fe256c6dc9b040ba66fc9bbee25349babff815fa6"
BASE = PROJECT_DIR / "artifacts/postgate/multi_nwp_consensus_g23_rescue_v1/validation/baseline_recent097_2024_h2.parquet"
BASE_SHA256 = "496687b38986f0a8933bd9d0d2404d1711de802b0e4e14d31e2ec4c2d00c7a55"
LABELS = Path(r"data/local/open/train/train_labels.csv")
LABELS_SHA256 = "47bb64252195cf4734e67394d6e50485f27a608def3b5a8791fcc7674bbceb03"
OUT = PROJECT_DIR / "artifacts/postgate/multi_nwp_disjoint_expert_g12joint_g3consensus_v2"
WINDOWS = {
    "H2": (pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2025-01-01 00:00")),
    "Q3": (pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2024-09-30 23:00")),
    "Q4": (pd.Timestamp("2024-10-01 00:00"), pd.Timestamp("2025-01-01 00:00")),
    "Jul": (pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2024-07-31 23:00")),
    "Aug": (pd.Timestamp("2024-08-01 00:00"), pd.Timestamp("2024-08-31 23:00")),
    "Sep": (pd.Timestamp("2024-09-01 00:00"), pd.Timestamp("2024-09-30 23:00")),
    "Oct": (pd.Timestamp("2024-10-01 00:00"), pd.Timestamp("2024-10-31 23:00")),
    "Nov": (pd.Timestamp("2024-11-01 00:00"), pd.Timestamp("2024-11-30 23:00")),
    "Dec": (pd.Timestamp("2024-12-01 00:00"), pd.Timestamp("2025-01-01 00:00")),
}


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise AssertionError(f"frozen input changed: {path}")


def _delta(actual: pd.DataFrame, base: pd.DataFrame, candidate: pd.DataFrame, *, cols: tuple[str, ...]) -> dict[str, float]:
    before = score_details(actual, base, target_cols=cols).as_dict()
    after = score_details(actual, candidate, target_cols=cols).as_dict()
    return {
        "total": float(after["total_score"] - before["total_score"]),
        "one_minus_nmae": float(after["one_minus_nmae"] - before["one_minus_nmae"]),
        "ficr": float(after["ficr"] - before["ficr"]),
    }


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"fresh output required: {OUT}")
    for path, digest in ((CONFIG, CONFIG_SHA256), (JOINT, JOINT_SHA256), (CONSENSUS, CONSENSUS_SHA256), (BASE, BASE_SHA256)):
        _verify(path, digest)
    base = pd.read_parquet(BASE)
    joint = pd.read_parquet(JOINT)
    consensus = pd.read_parquet(CONSENSUS)
    for frame in (joint, consensus):
        if not frame.index.equals(base.index) or tuple(frame.columns) != tuple(TARGET_COLS):
            raise AssertionError("source increment schema/index mismatch")
    candidate = base.copy()
    ownership = {
        "kpx_group_1": 0.25 * joint["kpx_group_1"],
        "kpx_group_2": 0.25 * joint["kpx_group_2"],
        "kpx_group_3": 0.15 * consensus["kpx_group_3"],
    }
    for group in TARGET_COLS:
        cap = CAPACITY_KWH[group]
        candidate[group] = np.clip(
            base[group].to_numpy(dtype=np.float64) / cap + ownership[group].to_numpy(dtype=np.float64),
            0.0,
            1.02,
        ) * cap
    strict._atomic_parquet(base, OUT / "confirmation/base_scale097_2024_h2.parquet")
    strict._atomic_parquet(candidate, OUT / "confirmation/candidate_disjoint_2024_h2.parquet")
    lock = OUT / "candidate_before_label_lock.json"
    strict._write_json(lock, {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister": describe_file(CONFIG),
        "joint_increment": describe_file(JOINT),
        "consensus_increment": describe_file(CONSENSUS),
        "base": describe_file(OUT / "confirmation/base_scale097_2024_h2.parquet"),
        "candidate": describe_file(OUT / "confirmation/candidate_disjoint_2024_h2.parquet"),
        "ownership": {"kpx_group_1": "0.25*joint", "kpx_group_2": "0.25*joint", "kpx_group_3": "0.15*consensus"},
        "label_values_read_before_lock": 0,
        "public_feedback_or_subset_values_read": 0,
    })
    _verify(LABELS, LABELS_SHA256)
    labels = strict._read_full_labels(LABELS).loc[base.index, list(TARGET_COLS)]
    mixed: dict[str, dict[str, float]] = {}
    groups: dict[str, dict[str, dict[str, float]]] = {g: {} for g in TARGET_COLS}
    for name, (start, end) in WINDOWS.items():
        idx = base.index[(base.index >= start) & (base.index <= end)]
        mixed[name] = _delta(labels.loc[idx], base.loc[idx], candidate.loc[idx], cols=tuple(TARGET_COLS))
        for group in TARGET_COLS:
            groups[group][name] = _delta(labels.loc[idx, [group]], base.loc[idx, [group]], candidate.loc[idx, [group]], cols=(group,))
    positive = sum(mixed[name]["total"] > 0.0 for name in WINDOWS)
    checks = {
        "each_group_h2_positive": all(groups[g]["H2"]["total"] > 0.0 for g in TARGET_COLS),
        "mixed_h2_total_positive": mixed["H2"]["total"] > 0.0,
        "mixed_h2_ficr_positive": mixed["H2"]["ficr"] > 0.0,
        "mixed_q3_floor": mixed["Q3"]["total"] >= -0.0005,
        "mixed_q4_floor": mixed["Q4"]["total"] >= -0.0005,
        "positive_mixed_segments_ge_6": positive >= 6,
    }
    passed = all(checks.values())
    strict._write_json(OUT / "confirmation_results.json", {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister": describe_file(CONFIG),
        "candidate_before_label_lock": describe_file(lock),
        "mixed": mixed,
        "groups": groups,
        "positive_mixed_segments": positive,
        "checks": checks,
        "promoted": passed,
        "conditional_2025_assembly_allowed": passed,
        "selection_unsafe_posthoc": True,
        "year_2025_values_read": 0,
        "csv_created": False,
    })
    print(json.dumps({"promoted": passed, "checks": checks, "positive": positive, "mixed": mixed, "group_h2": {g: groups[g]["H2"] for g in TARGET_COLS}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

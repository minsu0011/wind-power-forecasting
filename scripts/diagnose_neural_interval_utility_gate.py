"""Selection-unsafe Stage1-only diagnostic for interval-utility neural gating.

This script is deliberately not a candidate builder.  It reuses already-scored
2023/H2-2023 OOF artifacts, reads no 2024/2025 values, creates no preregister,
and may not promote or generate a submission.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_ficr_bayes_decision_strict as bayes
from scripts import run_raw_grid_wind_lgb as raw_protocol
from scripts import run_raw_spatiotemporal_attention as neural_protocol
from src.manifest import describe_file, sha256_file, utc_now, write_json_atomic
from src.metric import CAPACITY_KWH, TARGET_COLS


NEURAL_DIR = PROJECT_DIR / "artifacts/postgate/raw_spatiotemporal_attention_strict_v1"
INTERVAL_DIR = PROJECT_DIR / "artifacts/postgate/direct_interval_probability_strict_v1"
OUTPUT = PROJECT_DIR / "artifacts/audits/neural_interval_utility_gate_selection_unsafe_v1.json"
MARGINS: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01)
ACTIONS: Mapping[str, float] = {"direct": 1.0, "blend_w025": 0.025, "blend_w050": 0.05}
DISAGREEMENT_THRESHOLDS: tuple[float, ...] = (0.0, 0.08, 0.12, 0.16, 0.20)


def _interpolated_utility(surface: np.ndarray, action_cf: np.ndarray) -> np.ndarray:
    position = np.clip(np.asarray(action_cf, dtype=np.float64), 0.0, 1.02) * 100.0
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, 102)
    fraction = position - lower
    rows = np.arange(len(position))
    return (1.0 - fraction) * surface[rows, lower] + fraction * surface[rows, upper]


def _summarize_comparison(comparison: Mapping[str, Any]) -> dict[str, Any]:
    deltas = {name: float(record["delta"]) for name, record in comparison.items()}
    nmae = {
        name: float(record["candidate"]["one_minus_nmae"])
        - float(record["baseline"]["one_minus_nmae"])
        for name, record in comparison.items()
    }
    ficr = {
        name: float(record["candidate"]["ficr"])
        - float(record["baseline"]["ficr"])
        for name, record in comparison.items()
    }
    return {
        "deltas": deltas,
        "nmae_deltas": nmae,
        "ficr_deltas": ficr,
        "minimum": min(deltas.values()),
        "mean": float(np.mean(list(deltas.values()))),
        "full": deltas["full"],
        "positive_slices": sum(value > 0.0 for value in deltas.values()),
        "slice_count": len(deltas),
        "all_slices_strictly_positive": all(value > 0.0 for value in deltas.values()),
    }


def main() -> None:
    neural_manifest = NEURAL_DIR / "manifest.json"
    interval_manifest = INTERVAL_DIR / "manifest.json"
    neural_audit = PROJECT_DIR / "artifacts/audits/raw_spatiotemporal_attention_strict_v1_cross_audit.json"
    interval_audit = PROJECT_DIR / "artifacts/audits/direct_interval_probability_strict_v1_independent.json"
    raw_contract = json.loads(
        (PROJECT_DIR / "configs/raw_grid_wind_lgb_preregister_v1.json").read_text(encoding="utf-8")
    )
    labels, label_evidence = raw_protocol._read_label_prefix(
        Path(r"data/local/open"),
        raw_contract["physical_stage1_inputs"]["labels_stage1_score_prefix_after_lock"],
        TARGET_COLS,
    )
    interval_oof = INTERVAL_DIR / "oof"
    neural_pred = NEURAL_DIR / "predictions"
    groups: dict[str, Any] = {}
    for group in TARGET_COLS:
        fit_id = "g12_2022" if group in TARGET_COLS[:2] else "g3_through_2023h1"
        index = neural_protocol.YEAR_2023 if group in TARGET_COLS[:2] else neural_protocol.G3_H2
        raw_cf = pd.read_parquet(neural_pred / f"stage1__{fit_id}__raw_cf.parquet")[group]
        baseline_kwh = pd.read_parquet(
            neural_pred / f"stage1__{fit_id}__{group}__baseline.parquet"
        )[group]
        baseline_cf = baseline_kwh.to_numpy(dtype=np.float64) / CAPACITY_KWH[group]
        diagnostic = pd.read_parquet(
            interval_oof / f"stage1_{group}_action_diagnostics.parquet"
        )
        if not diagnostic.index.equals(index):
            raise AssertionError(f"interval/neural application index changed: {group}")
        if not np.array_equal(diagnostic["baseline_cf"].to_numpy(), baseline_cf):
            raise AssertionError(f"interval/neural baseline changed: {group}")
        surface_path = interval_oof / f"stage1_{group}_surfaces.npz"
        with np.load(surface_path) as payload:
            utility = payload["utility"].copy()
        baseline_u = _interpolated_utility(utility, baseline_cf)
        raw_values = raw_cf.to_numpy(dtype=np.float64)
        segments = neural_protocol._segments(group)
        action_results: dict[str, Any] = {}
        for action_id, blend_weight in ACTIONS.items():
            action_cf = baseline_cf + blend_weight * (raw_values - baseline_cf)
            utility_delta = _interpolated_utility(utility, action_cf) - baseline_u
            margin_results: dict[str, Any] = {}
            for margin in MARGINS:
                gate = utility_delta > margin
                gated_cf = np.where(gate, action_cf, baseline_cf)
                candidate = pd.Series(
                    np.clip(gated_cf, 0.0, 1.02) * CAPACITY_KWH[group],
                    index=index,
                    name=group,
                )
                comparison = bayes._comparison(
                    labels.loc[index, group], baseline_kwh, candidate, group, segments
                )
                summary = _summarize_comparison(comparison)
                summary.update(
                    {
                        "margin": margin,
                        "gate_rows": int(gate.sum()),
                        "gate_fraction": float(gate.mean()),
                        "utility_delta_quantiles": {
                            str(q): float(np.quantile(utility_delta, q))
                            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
                        },
                    }
                )
                margin_results[f"m{int(round(margin * 10000)):04d}"] = summary
            action_results[action_id] = {
                "blend_weight": blend_weight,
                "margins": margin_results,
            }

        # Preserve the earlier simple-disagreement exploration, explicitly as
        # selection-unsafe evidence only; it is not eligible for promotion.
        disagreement: dict[str, Any] = {}
        absolute_disagreement = np.abs(raw_values - baseline_cf)
        for threshold in DISAGREEMENT_THRESHOLDS:
            gate = absolute_disagreement >= threshold
            action_cf = baseline_cf + 0.025 * gate * (raw_values - baseline_cf)
            candidate = pd.Series(
                np.clip(action_cf, 0.0, 1.02) * CAPACITY_KWH[group], index=index, name=group
            )
            comparison = bayes._comparison(
                labels.loc[index, group], baseline_kwh, candidate, group, segments
            )
            summary = _summarize_comparison(comparison)
            summary.update(
                {
                    "threshold_cf": threshold,
                    "fixed_blend_weight": 0.025,
                    "gate_rows": int(gate.sum()),
                    "gate_fraction": float(gate.mean()),
                }
            )
            disagreement[f"t{int(round(threshold * 100)):02d}"] = summary
        groups[group] = {
            "application_start": index.min().isoformat(),
            "application_end": index.max().isoformat(),
            "rows": len(index),
            "interval_surface": describe_file(surface_path),
            "actions": action_results,
            "simple_disagreement_diagnostic": disagreement,
        }

    payload = {
        "schema_version": 1,
        "diagnostic_type": "neural_interval_utility_gate_selection_unsafe_v1",
        "created_utc": utc_now(),
        "status": "selection_unsafe_diagnostic_only_do_not_promote",
        "reason": "all margins, actions, and disagreement thresholds were evaluated after the same Stage1 labels had already been scored",
        "v2_preregister_created": False,
        "candidate_or_submission_created": False,
        "year_2024_value_bytes_read": 0,
        "year_2025_value_bytes_read": 0,
        "sample_value_bytes_read": 0,
        "public_or_scale_artifact_bytes_read": 0,
        "utility_evaluation": "linear interpolation of the audited direct-interval v1 0.01-CF utility surface at both baseline and neural/blended actions",
        "gate_formula": "apply action only where U(action)-U(baseline)>margin; otherwise exact baseline",
        "minimal_margin_grid": list(MARGINS),
        "action_family": dict(ACTIONS),
        "groups": groups,
        "provenance": {
            "neural_manifest": describe_file(neural_manifest),
            "interval_manifest": describe_file(interval_manifest),
            "neural_cross_audit": describe_file(neural_audit),
            "interval_independent_audit": describe_file(interval_audit),
        },
        "score_label_evidence": label_evidence,
    }
    payload = json.loads(json.dumps(payload, default=str))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT, payload)
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256_file(OUTPUT)}))


if __name__ == "__main__":
    main()

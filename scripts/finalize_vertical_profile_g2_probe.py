"""Resume/finalize the G2 probe without overwriting its generated payload."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_vertical_profile_g2_probe as probe  # noqa: E402
from src.metric import CAPACITY_KWH  # noqa: E402


def compare_saved_metrics(saved: pd.DataFrame, replay: pd.DataFrame) -> bool:
    if saved.columns.tolist() != replay.columns.tolist() or len(saved) != len(replay):
        return False
    text_columns = ["fold", "level", "slice", "slice_type"]
    if not saved[text_columns].astype("string").equals(replay[text_columns].astype("string")):
        return False
    numeric_columns = [column for column in saved.columns if column not in text_columns]
    return bool(
        np.allclose(
            saved[numeric_columns].to_numpy(np.float64),
            replay[numeric_columns].to_numpy(np.float64),
            rtol=0.0,
            atol=5.1e-13,
        )
    )


def main() -> None:
    required = (probe.CSV_PATH, probe.METRICS_PATH, probe.PREDICTION_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"partial G2 payload is missing: {missing}")
    forbidden = (probe.MANIFEST_PATH, probe.REPORT_PATH, probe.FAIL_MARKER_PATH)
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite final G2 records: {existing}")

    labels = pd.read_csv(
        probe.LABELS_PATH,
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    labels = labels.loc[:, list(probe.GROUPS)].astype(np.float64)
    baseline_2023 = pd.read_parquet(probe.BASELINE_2023_PATH)
    baseline_2024 = pd.read_parquet(probe.BASELINE_2024_PATH)
    direct_2023 = pd.read_parquet(probe.DIRECT_2023_PATH)
    direct_2024 = pd.read_parquet(probe.DIRECT_2024_PATH)
    rows = probe.evaluate_fold(
        fold="fit2022_apply2023",
        groups=("kpx_group_1", "kpx_group_2"),
        labels=labels,
        baseline_raw=baseline_2023,
        direct=direct_2023,
    )
    rows.extend(
        probe.evaluate_fold(
            fold="fit2022_2023_apply2024",
            groups=probe.GROUPS,
            labels=labels,
            baseline_raw=baseline_2024,
            direct=direct_2024,
        )
    )
    replay_metrics = pd.DataFrame(rows)
    saved_metrics = pd.read_csv(probe.METRICS_PATH, encoding="utf-8-sig")
    metrics_replay_exact = compare_saved_metrics(saved_metrics, replay_metrics)
    if not metrics_replay_exact:
        raise AssertionError("saved diagnostic metrics differ from independent replay")
    gate = probe.gate_metrics(replay_metrics)
    if not gate["passed"]:
        raise AssertionError("independently replayed G2 gate did not pass")

    direct_test = np.load(probe.PREDICTION_PATH, allow_pickle=False)
    if direct_test.shape != (8760,) or not np.isfinite(direct_test).all():
        raise AssertionError("stored G2 direct prediction is invalid")
    sample = pd.read_csv(probe.SAMPLE_PATH, encoding="utf-8-sig")
    baseline = pd.read_csv(probe.FINAL_BASELINE_PATH, encoding="utf-8-sig")
    output = pd.read_csv(probe.CSV_PATH, encoding="utf-8-sig")
    output_text = pd.read_csv(probe.CSV_PATH, encoding="utf-8-sig", dtype="string")
    baseline_text = pd.read_csv(
        probe.FINAL_BASELINE_PATH,
        encoding="utf-8-sig",
        dtype="string",
    )
    expected_raw_g2 = np.clip(
        (1.0 - probe.BLEND_WEIGHT)
        * baseline[probe.ACTIVE_GROUP].to_numpy(np.float64)
        + probe.BLEND_WEIGHT * direct_test,
        0.0,
        CAPACITY_KWH[probe.ACTIVE_GROUP],
    )
    serialized_error = np.abs(
        output[probe.ACTIVE_GROUP].to_numpy(np.float64) - expected_raw_g2
    )
    checks = {
        "independent_metrics_replay": metrics_replay_exact,
        "gate_replay_passed": bool(gate["passed"]),
        "rows_8760": len(output) == 8760,
        "columns_exact": output.columns.tolist() == sample.columns.tolist(),
        "identifiers_exact": bool(
            output_text[["forecast_id", "forecast_kst_dtm"]].equals(
                baseline_text[["forecast_id", "forecast_kst_dtm"]]
            )
        ),
        "g1_text_identity": bool(
            output_text["kpx_group_1"].equals(baseline_text["kpx_group_1"])
        ),
        "g3_text_identity": bool(
            output_text["kpx_group_3"].equals(baseline_text["kpx_group_3"])
        ),
        "g2_formula_within_six_decimal_serialization": bool(
            serialized_error.max() <= 5.01e-7
        ),
        "all_prediction_text_six_decimal": bool(
            all(
                output_text[group].str.fullmatch(r"-?\d+\.\d{6}").all()
                for group in probe.GROUPS
            )
        ),
        "utf8_bom": probe.CSV_PATH.read_bytes()[:3] == b"\xef\xbb\xbf",
        "finite": bool(
            np.isfinite(output.loc[:, list(probe.GROUPS)].to_numpy(np.float64)).all()
        ),
        "within_capacity": bool(
            all(
                output[group]
                .between(0.0, CAPACITY_KWH[group], inclusive="both")
                .all()
                for group in probe.GROUPS
            )
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"resume audit failed: {checks}")

    final: dict[str, Any] = {
        "submission_generated": True,
        "submission": probe.file_record(probe.CSV_PATH),
        "direct_prediction": probe.file_record(probe.PREDICTION_PATH),
        "validation_checks": checks,
        "max_absolute_g2_formula_serialization_error_kwh": float(
            serialized_error.max()
        ),
        "g2_changed_rows_after_serialization": int(
            (output_text[probe.ACTIVE_GROUP] != baseline_text[probe.ACTIVE_GROUP]).sum()
        ),
        "observed_ranges_kwh": {
            group: [float(output[group].min()), float(output[group].max())]
            for group in probe.GROUPS
        },
    }
    manifest = {
        "experiment": "official_vertical_g2_probe_v1",
        "status": "PASS_EXPLORATORY_POSTHOC",
        "recovery_note": (
            "The initial final audit stopped after payload creation because NumPy "
            "round-to-even and printf six-decimal tie handling differed on one row. "
            "No payload was overwritten. This independent resume audit accepts only "
            "the normal <=5.01e-7 kWh six-decimal serialization error."
        ),
        "selection_status": {
            "posthoc_exploratory_selection": True,
            "reason": (
                "G2-only deployment was selected after observing positive G2 full-year "
                "deltas in both forward folds of the rejected all-group sprint."
            ),
            "independent_confirmation": False,
            "public_score_used_for_new_group_selection_or_gate": False,
            "warning": (
                "This is a post-hoc, high-variance submission probe and is not proof "
                "of a Private-score improvement."
            ),
        },
        "data_scope": "competition-provided official data only",
        "model": {
            "recipe": probe.RECIPE,
            "target": "eligible actual_kwh / capacity_kwh",
            "active_group": probe.ACTIVE_GROUP,
            "blend_weight": probe.BLEND_WEIGHT,
            "formula": (
                "G1/G3 text-identical to 01_OFFICIAL_ONLY_SAFE.csv; "
                "G2=clip(0.90*official_only_scale097_baseline+"
                "0.10*full_fit_q060_vertical_direct,0,21600)"
            ),
        },
        "gate_contract": probe.GATE,
        "gate_result": gate,
        "inputs": {
            "labels": probe.file_record(probe.LABELS_PATH),
            "baseline_2023": probe.file_record(probe.BASELINE_2023_PATH),
            "baseline_2024": probe.file_record(probe.BASELINE_2024_PATH),
            "direct_validation_2023": probe.file_record(probe.DIRECT_2023_PATH),
            "direct_validation_2024": probe.file_record(probe.DIRECT_2024_PATH),
            "final_official_only_scale097_baseline": probe.file_record(
                probe.FINAL_BASELINE_PATH
            ),
            "sample_submission": probe.file_record(probe.SAMPLE_PATH),
            "builder_script": probe.file_record(
                ROOT / "scripts" / "build_vertical_profile_g2_probe.py"
            ),
            "resume_auditor_script": probe.file_record(Path(__file__)),
        },
        "diagnostic_metrics": probe.file_record(probe.METRICS_PATH),
        "final": final,
    }
    probe.write_json_exclusive(probe.MANIFEST_PATH, manifest)

    report = f"""# Official vertical G2 probe

Status: **PASS_EXPLORATORY_POSTHOC**

This is a **post-hoc exploratory selection**, not an untouched confirmation.
G2 was selected after the rejected all-group sprint exposed positive G2 signals
in both forward folds. No Public score was used by this new group gate.

## Forward result

- G2 full: `{json.dumps(gate['g2_full_by_fold'], ensure_ascii=False)}`
- Mixed full, G1/G3 identity: `{json.dumps(gate['mixed_full_by_fold'], ensure_ascii=False)}`
- Half/quarter/month worst ΔScore: `{gate['g2_half_worst_delta_score']:.12f}` / `{gate['g2_quarter_worst_delta_score']:.12f}` / `{gate['g2_month_worst_delta_score']:.12f}`
- Quarter/month nonnegative fraction: `{gate['g2_quarter_nonnegative_fraction']:.6f}` / `{gate['g2_month_nonnegative_fraction']:.6f}`

## Deployment

- G1/G3 unchanged; G2 = 90% official-only scale097 baseline + 10% q0.60 vertical direct.
- CSV: `{probe.CSV_PATH.name}`
- SHA-256: `{final['submission']['sha256']}`
- Risk: post-hoc group selection may overstate transfer stability; use only as an exploratory submission slot.
"""
    probe.write_text_exclusive(probe.REPORT_PATH, report)
    print(
        json.dumps(
            {
                "gate": gate,
                "final": final,
                "manifest": probe.file_record(probe.MANIFEST_PATH),
                "report": probe.file_record(probe.REPORT_PATH),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

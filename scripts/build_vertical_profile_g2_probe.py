"""Post-hoc exploratory G2-only vertical q0.60 probe.

The full-family sprint identified a repeated G2 signal after results were seen.
This script therefore labels the group selection as post-hoc exploratory.  It
independently recomputes the requested G2 and mixed forward metrics from the
stored raw validation predictions, applies the fixed gate, and emits a 2025 CSV
only if every gate condition passes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_vertical_profile_direct_sprint import (  # noqa: E402
    CURRENT_BASE_SCALE,
    FINAL_BASELINE_PATH,
    GROUPS,
    LABELS_PATH,
    RAW,
    SAMPLE_PATH,
    add_vertical_features,
    load_feature_pair,
    make_model,
)
from src.metric import CAPACITY_KWH, group_metrics, score_details  # noqa: E402


OUT = ROOT / "artifacts" / "final_submission_sprint_20260812" / "vertical_direct"
RECIPE = "lgb_q060_vertical"
ACTIVE_GROUP = "kpx_group_2"
BLEND_WEIGHT = 0.10
BASELINE_2023_PATH = ROOT / "artifacts" / "oof" / "dev2023_locked_v3.parquet"
BASELINE_2024_PATH = (
    ROOT / "artifacts" / "oof" / "gate2024_recent_v4_cf_fix_calibration_fit.parquet"
)
DIRECT_2023_PATH = (
    OUT / "validation_predictions" / f"{RECIPE}__fit2022_apply2023__standalone.parquet"
)
DIRECT_2024_PATH = (
    OUT
    / "validation_predictions"
    / f"{RECIPE}__fit2022_2023_apply2024__standalone.parquet"
)

CSV_PATH = OUT / "03_OFFICIAL_VERTICAL_G2_PROBE.csv"
METRICS_PATH = OUT / "g2_probe_slice_metrics.csv"
PREDICTION_PATH = OUT / "g2_probe_direct_2025.npy"
MANIFEST_PATH = OUT / "G2_PROBE_MANIFEST.json"
REPORT_PATH = OUT / "G2_PROBE_REPORT.md"
FAIL_MARKER_PATH = OUT / "G2_PROBE_NO_CSV.txt"

GATE = {
    "g2_full_total_strictly_positive_each_fold": 0.0,
    "g2_full_n_and_f_min_each_fold": -0.0005,
    "g2_half_worst_total": -0.001,
    "g2_quarter_worst_total": -0.0015,
    "g2_month_worst_total": -0.006,
    "g2_quarter_min_nonnegative_fraction": 0.60,
    "g2_month_min_nonnegative_fraction": 0.50,
    "mixed_full_total_strictly_positive_each_fold": 0.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_text_exclusive(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    with path.open("x", encoding=encoding, newline="") as handle:
        handle.write(content)


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    write_text_exclusive(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    month = index.month.to_numpy()
    result: dict[str, np.ndarray] = {"full": np.ones(len(index), dtype=bool)}
    result["H1"] = month <= 6
    result["H2"] = month >= 7
    for quarter in range(1, 5):
        result[f"Q{quarter}"] = ((month - 1) // 3 + 1) == quarter
    for value in range(1, 13):
        result[f"M{value:02d}"] = month == value
    return result


def kind(name: str) -> str:
    if name == "full":
        return "full"
    if name.startswith("H"):
        return "half"
    if name.startswith("Q"):
        return "quarter"
    return "month"


def competition_tuple(
    actual: pd.DataFrame,
    prediction: pd.DataFrame,
    groups: Iterable[str],
) -> tuple[float, float, float]:
    selected = tuple(groups)
    result = score_details(
        actual,
        prediction,
        target_cols=selected,
        capacities=CAPACITY_KWH,
    )
    return result.total_score, result.one_minus_nmae, result.ficr


def evaluate_fold(
    *,
    fold: str,
    groups: tuple[str, ...],
    labels: pd.DataFrame,
    baseline_raw: pd.DataFrame,
    direct: pd.DataFrame,
) -> list[dict[str, Any]]:
    index = baseline_raw.index
    actual = labels.loc[index, list(groups)]
    baseline = CURRENT_BASE_SCALE * baseline_raw.loc[index, list(groups)].astype(np.float64)
    for group in groups:
        baseline[group] = baseline[group].clip(0.0, CAPACITY_KWH[group])
    proposed = baseline.copy()
    proposed[ACTIVE_GROUP] = np.clip(
        (1.0 - BLEND_WEIGHT) * baseline[ACTIVE_GROUP]
        + BLEND_WEIGHT * direct.loc[index, ACTIVE_GROUP].astype(np.float64),
        0.0,
        CAPACITY_KWH[ACTIVE_GROUP],
    )
    rows: list[dict[str, Any]] = []
    for slice_name, mask in masks(index).items():
        ix = index[mask]
        base_group = group_metrics(
            actual.loc[ix, ACTIVE_GROUP],
            baseline.loc[ix, ACTIVE_GROUP],
            CAPACITY_KWH[ACTIVE_GROUP],
            group_name=ACTIVE_GROUP,
        )
        candidate_group = group_metrics(
            actual.loc[ix, ACTIVE_GROUP],
            proposed.loc[ix, ACTIVE_GROUP],
            CAPACITY_KWH[ACTIVE_GROUP],
            group_name=ACTIVE_GROUP,
        )
        base_group_score = 0.5 * (base_group.one_minus_nmae + base_group.ficr)
        candidate_group_score = 0.5 * (
            candidate_group.one_minus_nmae + candidate_group.ficr
        )
        rows.append(
            {
                "fold": fold,
                "level": "g2",
                "slice": slice_name,
                "slice_type": kind(slice_name),
                "base_score": base_group_score,
                "candidate_score": candidate_group_score,
                "delta_score": candidate_group_score - base_group_score,
                "delta_one_minus_nmae": candidate_group.one_minus_nmae
                - base_group.one_minus_nmae,
                "delta_ficr": candidate_group.ficr - base_group.ficr,
            }
        )

        base_mixed = competition_tuple(actual.loc[ix], baseline.loc[ix], groups)
        candidate_mixed = competition_tuple(actual.loc[ix], proposed.loc[ix], groups)
        rows.append(
            {
                "fold": fold,
                "level": "mixed",
                "slice": slice_name,
                "slice_type": kind(slice_name),
                "base_score": base_mixed[0],
                "candidate_score": candidate_mixed[0],
                "delta_score": candidate_mixed[0] - base_mixed[0],
                "delta_one_minus_nmae": candidate_mixed[1] - base_mixed[1],
                "delta_ficr": candidate_mixed[2] - base_mixed[2],
            }
        )
    return rows


def gate_metrics(table: pd.DataFrame) -> dict[str, Any]:
    g2 = table[table["level"] == "g2"]
    mixed = table[table["level"] == "mixed"]
    full = g2[g2["slice_type"] == "full"]
    half = g2[g2["slice_type"] == "half"]
    quarter = g2[g2["slice_type"] == "quarter"]
    month = g2[g2["slice_type"] == "month"]
    mixed_full = mixed[mixed["slice_type"] == "full"]
    checks = {
        "g2_full_total_strictly_positive_each_fold": bool(
            (full["delta_score"] > GATE["g2_full_total_strictly_positive_each_fold"]).all()
        ),
        "g2_full_n_and_f_each_fold": bool(
            (
                full[["delta_one_minus_nmae", "delta_ficr"]]
                >= GATE["g2_full_n_and_f_min_each_fold"]
            ).all().all()
        ),
        "g2_half_worst_total": bool(
            half["delta_score"].min() >= GATE["g2_half_worst_total"]
        ),
        "g2_quarter_worst_total": bool(
            quarter["delta_score"].min() >= GATE["g2_quarter_worst_total"]
        ),
        "g2_month_worst_total": bool(
            month["delta_score"].min() >= GATE["g2_month_worst_total"]
        ),
        "g2_quarter_sign_fraction": bool(
            (quarter["delta_score"] >= 0.0).mean()
            >= GATE["g2_quarter_min_nonnegative_fraction"]
        ),
        "g2_month_sign_fraction": bool(
            (month["delta_score"] >= 0.0).mean()
            >= GATE["g2_month_min_nonnegative_fraction"]
        ),
        "mixed_full_total_strictly_positive_each_fold": bool(
            (
                mixed_full["delta_score"]
                > GATE["mixed_full_total_strictly_positive_each_fold"]
            ).all()
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "g2_full_by_fold": full[
            ["fold", "delta_score", "delta_one_minus_nmae", "delta_ficr"]
        ].to_dict(orient="records"),
        "mixed_full_by_fold": mixed_full[
            ["fold", "delta_score", "delta_one_minus_nmae", "delta_ficr"]
        ].to_dict(orient="records"),
        "g2_half_worst_delta_score": float(half["delta_score"].min()),
        "g2_quarter_worst_delta_score": float(quarter["delta_score"].min()),
        "g2_month_worst_delta_score": float(month["delta_score"].min()),
        "g2_quarter_nonnegative_fraction": float((quarter["delta_score"] >= 0.0).mean()),
        "g2_month_nonnegative_fraction": float((month["delta_score"] >= 0.0).mean()),
    }


def main() -> None:
    targets = (
        CSV_PATH,
        METRICS_PATH,
        PREDICTION_PATH,
        MANIFEST_PATH,
        REPORT_PATH,
        FAIL_MARKER_PATH,
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing G2 probe outputs: {existing}")

    labels = pd.read_csv(
        LABELS_PATH,
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    labels = labels.loc[:, list(GROUPS)].astype(np.float64)
    baseline_2023 = pd.read_parquet(BASELINE_2023_PATH)
    baseline_2024 = pd.read_parquet(BASELINE_2024_PATH)
    direct_2023 = pd.read_parquet(DIRECT_2023_PATH)
    direct_2024 = pd.read_parquet(DIRECT_2024_PATH)

    rows = evaluate_fold(
        fold="fit2022_apply2023",
        groups=("kpx_group_1", "kpx_group_2"),
        labels=labels,
        baseline_raw=baseline_2023,
        direct=direct_2023,
    )
    rows.extend(
        evaluate_fold(
            fold="fit2022_2023_apply2024",
            groups=GROUPS,
            labels=labels,
            baseline_raw=baseline_2024,
            direct=direct_2024,
        )
    )
    metrics = pd.DataFrame(rows)
    gate = gate_metrics(metrics)

    # Diagnostic CSV is new and written with exclusive-create mode.
    metrics.to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12f",
        mode="x",
    )

    final: dict[str, Any] = {"submission_generated": False}
    if gate["passed"]:
        train_features, test_features, vertical_names = load_feature_pair(ACTIVE_GROUP)
        capacity = CAPACITY_KWH[ACTIVE_GROUP]
        y = labels[ACTIVE_GROUP].to_numpy(np.float64)
        eligible = np.isfinite(y) & (y >= 0.10 * capacity)
        model = make_model(RECIPE, ACTIVE_GROUP)
        model.fit(train_features.iloc[np.flatnonzero(eligible)], y[eligible] / capacity)
        direct_test = np.clip(
            np.asarray(model.predict(test_features), dtype=np.float64) * capacity,
            0.0,
            capacity,
        )
        with PREDICTION_PATH.open("xb") as handle:
            np.save(handle, direct_test, allow_pickle=False)

        sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
        baseline_final = pd.read_csv(FINAL_BASELINE_PATH, encoding="utf-8-sig")
        if not sample[["forecast_id", "forecast_kst_dtm"]].equals(
            baseline_final[["forecast_id", "forecast_kst_dtm"]]
        ):
            raise AssertionError("fixed baseline identifiers differ from sample")
        submission = baseline_final.copy()
        submission[ACTIVE_GROUP] = np.clip(
            (1.0 - BLEND_WEIGHT)
            * baseline_final[ACTIVE_GROUP].to_numpy(np.float64)
            + BLEND_WEIGHT * direct_test,
            0.0,
            capacity,
        )
        submission.to_csv(
            CSV_PATH,
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
            mode="x",
        )

        output_text = pd.read_csv(CSV_PATH, encoding="utf-8-sig", dtype="string")
        baseline_text = pd.read_csv(FINAL_BASELINE_PATH, encoding="utf-8-sig", dtype="string")
        output_numeric = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        expected_g2 = np.round(
            np.clip(
                (1.0 - BLEND_WEIGHT)
                * baseline_final[ACTIVE_GROUP].to_numpy(np.float64)
                + BLEND_WEIGHT * direct_test,
                0.0,
                capacity,
            ),
            6,
        )
        checks = {
            "rows_8760": len(output_text) == 8760,
            "columns_exact": output_text.columns.tolist() == sample.columns.tolist(),
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
            "g2_formula_after_six_decimal_serialization": bool(
                np.array_equal(
                    output_numeric[ACTIVE_GROUP].to_numpy(np.float64),
                    expected_g2,
                )
            ),
            "utf8_bom": CSV_PATH.read_bytes()[:3] == b"\xef\xbb\xbf",
            "finite": bool(
                np.isfinite(output_numeric.loc[:, list(GROUPS)].to_numpy(np.float64)).all()
            ),
            "within_capacity": bool(
                all(
                    output_numeric[group]
                    .between(0.0, CAPACITY_KWH[group], inclusive="both")
                    .all()
                    for group in GROUPS
                )
            ),
        }
        if not all(checks.values()):
            raise AssertionError(f"G2 probe CSV validation failed: {checks}")
        final = {
            "submission_generated": True,
            "submission": file_record(CSV_PATH),
            "direct_prediction": file_record(PREDICTION_PATH),
            "validation_checks": checks,
            "eligible_full_fit_rows": int(eligible.sum()),
            "base_feature_count": 612,
            "created_vertical_feature_count": len(vertical_names),
            "total_feature_count": train_features.shape[1],
            "g2_changed_rows_after_serialization": int(
                (
                    output_text[ACTIVE_GROUP]
                    != baseline_text[ACTIVE_GROUP]
                ).sum()
            ),
            "observed_ranges_kwh": {
                group: [
                    float(output_numeric[group].min()),
                    float(output_numeric[group].max()),
                ]
                for group in GROUPS
            },
        }
    else:
        write_text_exclusive(
            FAIL_MARKER_PATH,
            "G2-only exploratory probe failed at least one fixed gate; no submission CSV was generated.\n",
        )
        final = {
            "submission_generated": False,
            "reason": "at least one G2-only exploratory stability gate failed",
            "marker": file_record(FAIL_MARKER_PATH),
        }

    manifest = {
        "experiment": "official_vertical_g2_probe_v1",
        "status": "PASS_EXPLORATORY_POSTHOC" if gate["passed"] else "NO_GO",
        "selection_status": {
            "posthoc_exploratory_selection": True,
            "reason": (
                "G2-only deployment was selected after observing that the full-family "
                "q0.60 10% blend had positive G2 full-year deltas in both forward folds."
            ),
            "independent_confirmation": False,
            "public_score_used_for_new_group_selection_or_gate": False,
            "warning": (
                "This is not an untouched confirmation. Treat it as a high-variance probe, "
                "not as proof of Private improvement."
            ),
        },
        "data_scope": "competition-provided official data only",
        "model": {
            "recipe": RECIPE,
            "target": "eligible actual_kwh / capacity_kwh",
            "active_group": ACTIVE_GROUP,
            "blend_weight": BLEND_WEIGHT,
            "formula": (
                "G1/G3 = byte-text identity to 01_OFFICIAL_ONLY_SAFE.csv; "
                "G2 = clip(0.90*official_only_scale097_baseline + "
                "0.10*full_fit_q060_vertical_direct, 0, 21600)"
            ),
        },
        "gate_contract": GATE,
        "gate_result": gate,
        "inputs": {
            "labels": file_record(LABELS_PATH),
            "baseline_2023": file_record(BASELINE_2023_PATH),
            "baseline_2024": file_record(BASELINE_2024_PATH),
            "direct_validation_2023": file_record(DIRECT_2023_PATH),
            "direct_validation_2024": file_record(DIRECT_2024_PATH),
            "final_official_only_scale097_baseline": file_record(FINAL_BASELINE_PATH),
            "sample_submission": file_record(SAMPLE_PATH),
        },
        "diagnostic_metrics": file_record(METRICS_PATH),
        "final": final,
    }
    write_json_exclusive(MANIFEST_PATH, manifest)

    g2_full = gate["g2_full_by_fold"]
    mixed_full = gate["mixed_full_by_fold"]
    report = f"""# Official vertical G2 probe

Status: **{manifest['status']}**

This is a **post-hoc exploratory selection**, not an untouched confirmation.
G2 was selected only after the all-group sprint exposed positive G2 full-year
signals in both forward folds. No Public score was used by this new gate.

## Forward result

- G2 full: `{json.dumps(g2_full, ensure_ascii=False)}`
- Mixed full with G1/G3 identity: `{json.dumps(mixed_full, ensure_ascii=False)}`
- G2 half worst ΔScore: `{gate['g2_half_worst_delta_score']:.12f}`
- G2 quarter worst ΔScore: `{gate['g2_quarter_worst_delta_score']:.12f}`
- G2 month worst ΔScore: `{gate['g2_month_worst_delta_score']:.12f}`
- Quarter/month nonnegative fractions: `{gate['g2_quarter_nonnegative_fraction']:.6f}` / `{gate['g2_month_nonnegative_fraction']:.6f}`

## Deployment

- Formula: G1/G3 unchanged; G2 = 90% official-only scale097 baseline + 10% q0.60 vertical direct.
- Submission generated: `{final['submission_generated']}`
- Risk: post-hoc group selection can overstate transfer stability; use as an exploratory slot only.
"""
    write_text_exclusive(REPORT_PATH, report)

    print(
        json.dumps(
            {
                "gate": gate,
                "final": final,
                "manifest": file_record(MANIFEST_PATH),
                "report": file_record(REPORT_PATH),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

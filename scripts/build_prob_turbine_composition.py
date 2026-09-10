"""Mechanically compose the stable probabilistic and SCADA attack predictions.

No fit, coefficient search, candidate search, or metric-driven choice occurs
here.  The formula is fixed to

    clip(prob_stable_g1g3 + (turbine_scada_attack - corrected_v3))

and is diagnosed on exact 2024 OOF counterparts before the same arithmetic is
applied to their 2025 parquets.  The result remains attack-only because Q2 and
Q3 reverse against corrected-v3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_turbine_scada_power import (  # noqa: E402
    TEST_START,
    YEAR_2024_END,
    YEAR_2024_START,
    YEAR_2025_END,
    _atomic_csv,
    _atomic_json,
    _atomic_parquet,
)
from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402


FORMULA = "clip(prob_stable_g1g3 + (turbine_scada_attack - corrected_v3), 0, 1.02*group_capacity)"
EXPECTED_SOURCE_SHA256 = {
    "prob_2024": "f0449955a62e8312e4dcaf200228633c3564c56601524cf416967a1bd3ff9875",
    "turbine_2024": "f4dce8184284e6c2b53d008e3ca8725647138406da14bf6b491ebe66c4b09402",
    "corrected_v3_2024": "9ff992c2d3fa3ce0725600fda415f98bd9a9b114cf6cb259df6f0b796005b76a",
    "prob_2025": "575cfe69fd216b534939579173201dcd963628f23898575e6b4a8d904564f1f3",
    "turbine_2025": "71fe9a493494b3f134ef53410633b7f51c631d5b5b07d254ff138137a25c6cb8",
    "corrected_v3_2025": "3d5d9c302f9e09bcad548548f9e3e550e92f90d5cd4493815b27f540aa9fd363",
    "labels": "47bb64252195cf4734e67394d6e50485f27a608def3b5a8791fcc7674bbceb03",
    "sample": "c925d2066a834f937f8091ed55acfe50ff86c8be4745b52c3adc95b056c5aaaa",
}
EXPECTED_2024_FULL_SCORE = 0.6433292298239777
EXPECTED_2024_FULL_DELTA_VS_CORRECTED_V3 = 0.003079035498573579
H2_2024_START = pd.Timestamp("2024-07-01 01:00:00")
Q2_2024_START = pd.Timestamp("2024-04-01 01:00:00")
Q3_2024_START = H2_2024_START
Q4_2024_START = pd.Timestamp("2024-10-01 01:00:00")
SLICES = {
    "full": (YEAR_2024_START, YEAR_2024_END),
    "H1": (YEAR_2024_START, H2_2024_START - pd.Timedelta(hours=1)),
    "H2": (H2_2024_START, YEAR_2024_END),
    "Q1": (YEAR_2024_START, Q2_2024_START - pd.Timedelta(hours=1)),
    "Q2": (Q2_2024_START, Q3_2024_START - pd.Timedelta(hours=1)),
    "Q3": (Q3_2024_START, Q4_2024_START - pd.Timedelta(hours=1)),
    "Q4": (Q4_2024_START, YEAR_2024_END),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/prob_turbine_composition"),
    )
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.artifact_root
    return {
        "prob_2024": root
        / "postgate"
        / "probabilistic_ficr"
        / "oof"
        / "stable_g1g3_transfer_2023_to_2024.parquet",
        "turbine_2024": root
        / "postgate"
        / "turbine_scada_power"
        / "oof"
        / "stage2_locked_candidates.parquet",
        "corrected_v3_2024": root / "oof" / "gate2024_locked_v3_cf_fix.parquet",
        "prob_2025": root
        / "postgate"
        / "probabilistic_ficr"
        / "predictions"
        / "prob_stable_g1g3_2025_test.parquet",
        "turbine_2025": root
        / "postgate"
        / "turbine_scada_power_attack"
        / "predictions"
        / "corrected_v3_g1_scada_attack_2025.parquet",
        "corrected_v3_2025": root
        / "final_cf_fix"
        / "predictions"
        / "corrected_v3_test.parquet",
        "labels": args.raw_dir / "train" / "train_labels.csv",
        "sample": args.raw_dir / "sample_submission.csv",
    }


def _verify_source_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != EXPECTED_SOURCE_SHA256:
        changed = {
            name: {"expected": EXPECTED_SOURCE_SHA256.get(name), "observed": digest}
            for name, digest in observed.items()
            if EXPECTED_SOURCE_SHA256.get(name) != digest
        }
        raise AssertionError(f"composition source identity changed: {changed}")
    return observed


def _read_prediction(path: Path, expected_index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGET_COLS:
        raise AssertionError(f"{path} target schema changed")
    if not frame.index.equals(expected_index):
        raise AssertionError(f"{path} index changed")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise AssertionError(f"{path} contains non-finite predictions")
    return frame.astype(float)


def _compose(
    prob: pd.DataFrame, turbine: pd.DataFrame, corrected_v3: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not prob.index.equals(turbine.index) or not prob.index.equals(corrected_v3.index):
        raise AssertionError("composition index mismatch")
    delta = turbine - corrected_v3
    if not np.array_equal(
        delta["kpx_group_2"].to_numpy(dtype=float), np.zeros(len(delta), dtype=float)
    ):
        raise AssertionError("turbine delta must leave group 2 unchanged")
    if not np.array_equal(
        delta["kpx_group_3"].to_numpy(dtype=float), np.zeros(len(delta), dtype=float)
    ):
        raise AssertionError("turbine delta must leave group 3 unchanged")
    result = prob + delta
    clip_counts: dict[str, int] = {}
    for group in TARGET_COLS:
        before = result[group].to_numpy(dtype=float, copy=True)
        after = np.clip(before, 0.0, 1.02 * CAPACITY_KWH[group])
        clip_counts[group] = int(np.count_nonzero(after != before))
        result[group] = after
    if not np.array_equal(
        result["kpx_group_2"].to_numpy(dtype=float),
        prob["kpx_group_2"].to_numpy(dtype=float),
    ):
        raise AssertionError("composed group 2 must be bit-identical to prob_stable")
    if not np.array_equal(
        result["kpx_group_3"].to_numpy(dtype=float),
        prob["kpx_group_3"].to_numpy(dtype=float),
    ):
        raise AssertionError("composed group 3 must be bit-identical to prob_stable")
    return result, {
        "formula": FORMULA,
        "clip_counts": clip_counts,
        "turbine_delta_nonzero_rows": {
            group: int(np.count_nonzero(delta[group].to_numpy(dtype=float)))
            for group in TARGET_COLS
        },
        "group2_unchanged_from_prob_stable_bit_exact": True,
        "group3_unchanged_from_prob_stable_bit_exact": True,
    }


def _score_summary(labels: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    metric = score_details(labels, prediction)
    return {
        "total_score": float(metric.total_score),
        "one_minus_nmae": float(metric.one_minus_nmae),
        "ficr": float(metric.ficr),
        "by_group": {
            group: {
                "score": float(
                    0.5
                    * (
                        values.one_minus_nmae
                        + values.ficr
                    )
                ),
                "one_minus_nmae": float(values.one_minus_nmae),
                "ficr": float(values.ficr),
                "n_evaluated": int(values.n_evaluated),
            }
            for group, values in metric.by_group.items()
        },
    }


def _diagnostics(
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    prob: pd.DataFrame,
    composed: pd.DataFrame,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, (start, end) in SLICES.items():
        index = labels.index[(labels.index >= start) & (labels.index <= end)]
        baseline_score = _score_summary(labels.loc[index], baseline.loc[index])
        prob_score = _score_summary(labels.loc[index], prob.loc[index])
        composed_score = _score_summary(labels.loc[index], composed.loc[index])
        output[name] = {
            "corrected_v3": baseline_score,
            "prob_stable_g1g3": prob_score,
            "composition": composed_score,
            "delta_vs_corrected_v3": float(
                composed_score["total_score"] - baseline_score["total_score"]
            ),
            "delta_vs_prob_stable_g1g3": float(
                composed_score["total_score"] - prob_score["total_score"]
            ),
        }
    full = output["full"]
    if abs(full["composition"]["total_score"] - EXPECTED_2024_FULL_SCORE) > 1e-15:
        raise AssertionError("2024 full composition score did not reproduce exactly")
    if (
        abs(
            full["delta_vs_corrected_v3"]
            - EXPECTED_2024_FULL_DELTA_VS_CORRECTED_V3
        )
        > 1e-15
    ):
        raise AssertionError("2024 full delta did not reproduce exactly")
    if not all(output[name]["delta_vs_corrected_v3"] > 0.0 for name in ("full", "H1", "H2")):
        raise AssertionError("full/H1/H2 positive transfer contract changed")
    if not (
        output["Q2"]["delta_vs_corrected_v3"] < 0.0
        and output["Q3"]["delta_vs_corrected_v3"] < 0.0
    ):
        raise AssertionError("expected Q2/Q3 reversal is absent")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.raw_dir = args.raw_dir.expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"composition output must be new and empty: {args.out_dir}")
    source_paths = _source_paths(args)
    source_hashes = _verify_source_hashes(source_paths)

    index_2024 = pd.date_range(
        YEAR_2024_START, YEAR_2024_END, freq="h", name="forecast_kst_dtm"
    )
    prob_2024 = _read_prediction(source_paths["prob_2024"], index_2024)
    turbine_2024 = _read_prediction(source_paths["turbine_2024"], index_2024)
    baseline_2024 = _read_prediction(source_paths["corrected_v3_2024"], index_2024)
    composition_2024, composition_audit_2024 = _compose(
        prob_2024, turbine_2024, baseline_2024
    )
    labels = pd.read_csv(source_paths["labels"], encoding="utf-8-sig")
    if tuple(labels.columns) != ("kst_dtm", *TARGET_COLS):
        raise AssertionError("label schema changed")
    labels.index = pd.DatetimeIndex(
        pd.to_datetime(labels.pop("kst_dtm"), errors="raise"),
        name="forecast_kst_dtm",
    )
    labels_2024 = labels.loc[index_2024, list(TARGET_COLS)].astype(float)
    diagnostic = _diagnostics(
        labels_2024, baseline_2024, prob_2024, composition_2024
    )

    sample = pd.read_csv(source_paths["sample"], encoding="utf-8-sig")
    expected_columns = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
    if tuple(sample.columns) != expected_columns or len(sample) != 8760:
        raise AssertionError("sample schema/row count changed")
    index_2025 = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if (
        index_2025.min() != TEST_START
        or index_2025.max() != YEAR_2025_END
        or not index_2025.is_unique
        or not index_2025.is_monotonic_increasing
    ):
        raise AssertionError("sample time contract changed")
    prob_2025 = _read_prediction(source_paths["prob_2025"], index_2025)
    turbine_2025 = _read_prediction(source_paths["turbine_2025"], index_2025)
    baseline_2025 = _read_prediction(source_paths["corrected_v3_2025"], index_2025)
    composition_2025, composition_audit_2025 = _compose(
        prob_2025, turbine_2025, baseline_2025
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.out_dir / "oof" / "prob_turbine_composition_2024.parquet"
    prediction_path = (
        args.out_dir / "predictions" / "prob_turbine_composition_2025.parquet"
    )
    submission_path = args.out_dir / "prob_turbine_composition_2025.csv"
    _atomic_parquet(composition_2024, oof_path)
    _atomic_parquet(composition_2025, prediction_path)
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = composition_2025[group].to_numpy(dtype=float)
    _atomic_csv(submission, submission_path)

    oof_readback = pd.read_parquet(oof_path, engine="pyarrow")
    prediction_readback = pd.read_parquet(prediction_path, engine="pyarrow")
    if (
        not oof_readback.index.equals(index_2024)
        or tuple(oof_readback.columns) != TARGET_COLS
        or not np.array_equal(
        oof_readback.to_numpy(dtype=float), composition_2024.to_numpy(dtype=float)
        )
    ):
        raise AssertionError("2024 OOF parquet readback changed values")
    if (
        not prediction_readback.index.equals(index_2025)
        or tuple(prediction_readback.columns) != TARGET_COLS
        or not np.array_equal(
        prediction_readback.to_numpy(dtype=float), composition_2025.to_numpy(dtype=float)
        )
    ):
        raise AssertionError("2025 prediction parquet readback changed values")
    if not submission_path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise AssertionError("submission UTF-8 BOM missing")
    csv_readback = pd.read_csv(submission_path, encoding="utf-8-sig")
    if tuple(csv_readback.columns) != expected_columns or len(csv_readback) != 8760:
        raise AssertionError("submission readback schema/row count changed")
    if not csv_readback["forecast_id"].equals(sample["forecast_id"]):
        raise AssertionError("submission IDs changed")
    if not csv_readback["forecast_kst_dtm"].equals(sample["forecast_kst_dtm"]):
        raise AssertionError("submission timestamps changed")
    csv_max_abs = 0.0
    for group in TARGET_COLS:
        expected = composition_2025[group].to_numpy(dtype=float)
        observed = csv_readback[group].to_numpy(dtype=float)
        difference = float(np.max(np.abs(observed - expected)))
        csv_max_abs = max(csv_max_abs, difference)
        if difference > 5.01e-7 or not np.isfinite(observed).all():
            raise AssertionError(f"{group} CSV readback failed: {difference}")
        if float(observed.min()) < 0.0 or float(observed.max()) > 1.02 * CAPACITY_KWH[group] + 1e-9:
            raise AssertionError(f"{group} CSV capacity clipping failed")

    results = {
        "schema_version": 1,
        "artifact_type": "prob_turbine_mechanical_composition",
        "formula": FORMULA,
        "no_fit": True,
        "no_coefficient_search": True,
        "no_candidate_search": True,
        "selection_unsafe": True,
        "strict_adoption": False,
        "attack_submission_only": True,
        "strict_rejection_reason": "2024 full/H1/H2 improve, but Q2 and Q3 are negative versus corrected-v3",
        "source_hashes": source_hashes,
        "diagnostic_2024": diagnostic,
        "composition_audit_2024": composition_audit_2024,
        "composition_audit_2025": composition_audit_2025,
        "verification": {
            "oof_parquet_readback_bit_exact": True,
            "prediction_parquet_readback_bit_exact": True,
            "group2_unchanged_from_prob_stable_bit_exact": True,
            "group3_unchanged_from_prob_stable_bit_exact": True,
            "sample_schema_exact": True,
            "sample_id_exact": True,
            "sample_timestamp_exact": True,
            "utf8_bom": True,
            "finite": True,
            "capacity_clip": True,
            "csv_max_abs_roundtrip_difference": csv_max_abs,
        },
        "artifacts": {
            "oof": str(oof_path.resolve()),
            "oof_sha256": sha256_file(oof_path),
            "prediction": str(prediction_path.resolve()),
            "prediction_sha256": sha256_file(prediction_path),
            "submission": str(submission_path.resolve()),
            "submission_sha256": sha256_file(submission_path),
        },
        "leaderboard_score_claim": False,
    }
    results_path = args.out_dir / "results.json"
    _atomic_json(results_path, _json_ready(results))

    provenance_paths = [
        Path(__file__).resolve(),
        PROJECT_DIR / "scripts" / "run_turbine_scada_power.py",
        args.artifact_root / "postgate" / "probabilistic_ficr" / "results.json",
        args.artifact_root / "postgate" / "turbine_scada_power" / "stage2_results.json",
        args.artifact_root / "postgate" / "turbine_scada_power_attack" / "results.json",
    ]
    output_files = [oof_path, prediction_path, submission_path, results_path]
    manifest = make_manifest(
        artifact_type="prob_turbine_mechanical_composition",
        parameters={
            "formula": FORMULA,
            "no_fit": True,
            "no_coefficient_search": True,
            "selection_unsafe": True,
            "strict_adoption": False,
        },
        input_files=[*source_paths.values(), *provenance_paths],
        output_files=output_files,
        results={
            "diagnostic_2024_full_score": diagnostic["full"]["composition"]["total_score"],
            "diagnostic_2024_full_delta_vs_corrected_v3": diagnostic["full"]["delta_vs_corrected_v3"],
            "quarter_deltas_vs_corrected_v3": {
                name: values["delta_vs_corrected_v3"]
                for name, values in diagnostic.items()
            },
            "submission_sha256": results["artifacts"]["submission_sha256"],
            "selection_unsafe": True,
            "strict_adoption": False,
            "leaderboard_score_claim": False,
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(args.out_dir / "manifest.json", manifest, overwrite=False)
    print(
        json.dumps(
            {
                "submission": str(submission_path),
                "sha256": results["artifacts"]["submission_sha256"],
                "diagnostic_2024_full_score": diagnostic["full"]["composition"]["total_score"],
                "delta_vs_corrected_v3": diagnostic["full"]["delta_vs_corrected_v3"],
                "selection_unsafe": True,
                "strict_adoption": False,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

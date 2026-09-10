"""Audit a predeclared spatial-diversity variant of the g3 threshold meta model.

The spatial predictions are *features only*.  They never replace the locked
base forecast or enter the decision candidate grid directly.  The variant is
first compared with the otherwise identical threshold-meta model on disjoint
2023 Q3<->Q4 fits.  It is then transferred unchanged from all available 2023
g3 OOF rows to 2024 and checked separately on both halves.  A 2025 candidate
is written only when every predeclared stability gate is positive.

The g3 core-Q07 model is intentionally excluded because no promoted full-2025
prediction exists for it.  The common, deployable feature set is core L1,
sequence L1, and sequence Q07.  In particular, the weaker 400-tree spatial
Q07 prediction is used only as a conditional diversity feature, never as a
standalone replacement.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_probabilistic_ficr import (  # noqa: E402
    COMMON_CANDIDATES,
    _assert_outputs_available,
    _atomic_joblib,
    _atomic_parquet,
    _atomic_submission,
    _gate_2024_sources,
    _g3_2023_sources,
    _group_summary,
    _interval,
    _meta_features,
    _read_labels,
    _test_2025_sources,
    _verify_submission,
)
from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402
from src.probabilistic import (  # noqa: E402
    ProbabilisticDecisionConfig,
    ProbabilisticFICRDecision,
)


GROUP = "kpx_group_3"
SPATIAL_FEATURES = (
    "spatial_core_l1_cf",
    "spatial_sequence_l1_cf",
    "spatial_sequence_q07_cf",
)
SPATIAL_2023_COLUMNS = {
    "spatial_core_l1_cf": "kpx_group_3__v1_plus_v2_core__l1",
    "spatial_sequence_l1_cf": "kpx_group_3__v1_plus_v2_seq__l1",
    "spatial_sequence_q07_cf": "kpx_group_3__v1_plus_v2_seq__q07",
}
SPATIAL_2024_COLUMNS = {
    "spatial_core_l1_cf": "candidate__kpx_group_3__core__l1",
    "spatial_sequence_l1_cf": "candidate__kpx_group_3__sequence__l1",
    "spatial_sequence_q07_cf": "candidate__kpx_group_3__sequence__q07",
}
SPATIAL_2025_COLUMNS = {
    "spatial_core_l1_cf": "kpx_group_3__core__l1",
    "spatial_sequence_l1_cf": "kpx_group_3__sequence__l1",
    "spatial_sequence_q07_cf": "kpx_group_3__sequence__q07",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(r"data/local/open"),
    )
    parser.add_argument("--oof-dir", type=Path, default=Path("artifacts/oof"))
    parser.add_argument("--gate-dir", type=Path, default=Path("artifacts/gate/v3"))
    parser.add_argument("--final-dir", type=Path, default=Path("artifacts/final_v3"))
    parser.add_argument(
        "--corrected-dir", type=Path, default=Path("artifacts/final_cf_fix")
    )
    parser.add_argument(
        "--spatial-2023",
        type=Path,
        default=Path(
            "artifacts/experiments/spatial_v2_seq/spatial_v2_dev_oof.parquet"
        ),
    )
    parser.add_argument(
        "--spatial-2024",
        type=Path,
        default=Path(
            "artifacts/experiments/spatial_v2_postgate/"
            "spatial_v2_2024_oof.parquet"
        ),
    )
    parser.add_argument(
        "--spatial-2025",
        type=Path,
        default=Path(
            "artifacts/experiments/spatial_v2_postgate/"
            "spatial_v2_promoted_2025_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--stable-prob-dir",
        type=Path,
        default=Path("artifacts/postgate/probabilistic_ficr"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/probabilistic_spatial_g3"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _spatial_features(
    path: Path,
    index: pd.DatetimeIndex,
    columns: Mapping[str, str],
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{path} must have a DatetimeIndex")
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path} index must be unique and sorted")
    missing_columns = sorted(set(columns.values()).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"{path} missing spatial columns: {missing_columns}")
    missing_rows = index.difference(frame.index)
    if len(missing_rows):
        raise ValueError(f"{path} missing {len(missing_rows)} requested timestamps")
    result = frame.loc[index, list(columns.values())].rename(
        columns={source: target for target, source in columns.items()}
    )
    result = result.loc[:, list(SPATIAL_FEATURES)].astype(float)
    result /= CAPACITY_KWH[GROUP]
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{path} has non-finite spatial features on target rows")
    return result


def _fit_apply(
    features: pd.DataFrame,
    actual: pd.Series,
    base: pd.Series,
    fit_index: pd.DatetimeIndex,
    apply_index: pd.DatetimeIndex,
    config: ProbabilisticDecisionConfig,
) -> tuple[pd.Series, ProbabilisticFICRDecision]:
    overlap = fit_index.intersection(apply_index)
    if len(overlap):
        raise AssertionError(f"fit/application overlap: {len(overlap)} rows")
    decision = ProbabilisticFICRDecision("threshold_meta", config=config).fit(
        features.loc[fit_index],
        actual.loc[fit_index],
        base.loc[fit_index],
        capacity_kwh=CAPACITY_KWH[GROUP],
    )
    prediction = decision.predict(features.loc[apply_index], base.loc[apply_index])
    return prediction, decision


def _comparison(
    actual: pd.Series,
    base_prediction: pd.Series,
    spatial_prediction: pd.Series,
) -> dict[str, Any]:
    baseline = _group_summary(actual, base_prediction, GROUP)
    spatial = _group_summary(actual, spatial_prediction, GROUP)
    return {
        "baseline_threshold_meta": baseline,
        "spatial_feature_threshold_meta": spatial,
        "delta_vs_baseline_threshold_meta": spatial["score"] - baseline["score"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    labels = _read_labels(args.raw_dir)
    candidates_2023, base_2023, inputs_2023 = _g3_2023_sources(args.oof_dir)
    candidates_2024, base_2024, inputs_2024 = _gate_2024_sources(
        args.gate_dir, args.oof_dir
    )
    candidates_2025, base_2025, sample, inputs_2025 = _test_2025_sources(
        args.raw_dir, args.final_dir, args.corrected_dir
    )
    candidates_2023 = {name: candidates_2023[name] for name in COMMON_CANDIDATES}
    candidates_2024 = {name: candidates_2024[name] for name in COMMON_CANDIDATES}
    candidates_2025 = {name: candidates_2025[name] for name in COMMON_CANDIDATES}

    base_features_2023 = _meta_features(candidates_2023, base_2023, GROUP)
    base_features_2024 = _meta_features(candidates_2024, base_2024, GROUP)
    base_features_2025 = _meta_features(candidates_2025, base_2025, GROUP)
    spatial_2023 = _spatial_features(
        args.spatial_2023, base_features_2023.index, SPATIAL_2023_COLUMNS
    )
    spatial_2024 = _spatial_features(
        args.spatial_2024, base_features_2024.index, SPATIAL_2024_COLUMNS
    )
    spatial_2025 = _spatial_features(
        args.spatial_2025, base_features_2025.index, SPATIAL_2025_COLUMNS
    )
    augmented_2023 = pd.concat([base_features_2023, spatial_2023], axis=1)
    augmented_2024 = pd.concat([base_features_2024, spatial_2024], axis=1)
    augmented_2025 = pd.concat([base_features_2025, spatial_2025], axis=1)
    expected_extra = tuple(augmented_2023.columns[-len(SPATIAL_FEATURES) :])
    if expected_extra != SPATIAL_FEATURES:
        raise AssertionError("spatial feature order changed")
    if tuple(augmented_2023.columns) != tuple(augmented_2024.columns) or tuple(
        augmented_2023.columns
    ) != tuple(augmented_2025.columns):
        raise AssertionError("train/transfer/test feature schemas differ")

    q3 = _interval("2023-07-01 01:00:00", "2023-10-01 00:00:00")
    q4 = _interval("2023-10-01 01:00:00", "2024-01-01 00:00:00")
    h1_2024 = _interval("2024-01-01 01:00:00", "2024-07-01 00:00:00")
    h2_2024 = _interval("2024-07-01 01:00:00", "2025-01-01 00:00:00")
    actual_2023 = labels.loc[base_2023.index, GROUP]
    actual_2024 = labels.loc[base_2024.index, GROUP]
    if actual_2023.isna().any():
        raise ValueError("2023 H2 g3 labels are incomplete")
    # Six known 2024 g3 labels are missing.  The official metric and the
    # decision fitter both exclude non-finite actuals while retaining the
    # complete hourly feature/prediction index.
    missing_2024_labels = int(actual_2024.isna().sum())

    print(
        "validated schemas: "
        f"2023={augmented_2023.shape}, 2024={augmented_2024.shape}, "
        f"2025={augmented_2025.shape}"
    )
    if args.dry_run:
        print("dry-run complete: no model fit, score, or output write")
        return 0

    paths = {
        "crossfit_baseline": args.out_dir / "oof/crossfit_2023_threshold_base.parquet",
        "crossfit_spatial": args.out_dir / "oof/crossfit_2023_threshold_spatial.parquet",
        "transfer_baseline": args.out_dir / "oof/transfer_2023_to_2024_threshold_base.parquet",
        "transfer_spatial": args.out_dir / "oof/transfer_2023_to_2024_threshold_spatial.parquet",
        "prediction": args.out_dir / "predictions/spatial_g3_2025_test.parquet",
        "candidate": args.out_dir / "prob_stable_g1g3_spatial_2025.csv",
        "model": args.out_dir / "models/spatial_g3_2024_fit.joblib",
        "results": args.out_dir / "results.json",
        "manifest": args.out_dir / "manifest.json",
    }
    # Candidate/model/prediction are conditional.  Existing diagnostic outputs
    # still require an explicit overwrite to keep experiment identity stable.
    _assert_outputs_available(
        [
            paths["crossfit_baseline"],
            paths["crossfit_spatial"],
            paths["transfer_baseline"],
            paths["transfer_spatial"],
            paths["results"],
            paths["manifest"],
        ],
        bool(args.overwrite),
    )

    config = ProbabilisticDecisionConfig(random_state=args.seed)
    crossfit_base = base_2023.copy()
    crossfit_spatial = base_2023.copy()
    direction_results: dict[str, Any] = {}
    for name, fit_index, apply_index in (
        ("q3_fit_q4_apply", q3, q4),
        ("q4_fit_q3_apply", q4, q3),
    ):
        print(name, flush=True)
        baseline_prediction, baseline_model = _fit_apply(
            base_features_2023,
            actual_2023,
            base_2023[GROUP],
            fit_index,
            apply_index,
            config,
        )
        spatial_prediction, spatial_model = _fit_apply(
            augmented_2023,
            actual_2023,
            base_2023[GROUP],
            fit_index,
            apply_index,
            config,
        )
        crossfit_base.loc[apply_index, GROUP] = baseline_prediction
        crossfit_spatial.loc[apply_index, GROUP] = spatial_prediction
        direction_results[name] = {
            **_comparison(
                actual_2023.loc[apply_index],
                baseline_prediction,
                spatial_prediction,
            ),
            "fit_rows": len(fit_index),
            "application_rows": len(apply_index),
            "fit_application_overlap": len(fit_index.intersection(apply_index)),
            "baseline_metadata": baseline_model.metadata(),
            "spatial_metadata": spatial_model.metadata(),
        }

    selected_on_2023 = all(
        result["delta_vs_baseline_threshold_meta"] > 0.0
        for result in direction_results.values()
    )

    # Fixed transfer: both variants fit exactly once on all 2023 H2 OOF rows.
    baseline_transfer_model = ProbabilisticFICRDecision(
        "threshold_meta", config=config
    ).fit(
        base_features_2023,
        actual_2023,
        base_2023[GROUP],
        capacity_kwh=CAPACITY_KWH[GROUP],
    )
    spatial_transfer_model = ProbabilisticFICRDecision(
        "threshold_meta", config=config
    ).fit(
        augmented_2023,
        actual_2023,
        base_2023[GROUP],
        capacity_kwh=CAPACITY_KWH[GROUP],
    )
    baseline_transfer = baseline_transfer_model.predict(
        base_features_2024, base_2024[GROUP]
    )
    spatial_transfer = spatial_transfer_model.predict(
        augmented_2024, base_2024[GROUP]
    )
    transfer_results = {
        period: _comparison(
            actual_2024.loc[index],
            baseline_transfer.loc[index],
            spatial_transfer.loc[index],
        )
        for period, index in (
            ("full", actual_2024.index),
            ("h1", h1_2024),
            ("h2", h2_2024),
        )
    }
    confirmed_on_2024 = all(
        transfer_results[period]["delta_vs_baseline_threshold_meta"] > 0.0
        for period in ("h1", "h2")
    )
    adopted = bool(selected_on_2023 and confirmed_on_2024)

    _atomic_parquet(crossfit_base, paths["crossfit_baseline"])
    _atomic_parquet(crossfit_spatial, paths["crossfit_spatial"])
    _atomic_parquet(
        pd.DataFrame({GROUP: baseline_transfer}, index=base_2024.index),
        paths["transfer_baseline"],
    )
    _atomic_parquet(
        pd.DataFrame({GROUP: spatial_transfer}, index=base_2024.index),
        paths["transfer_spatial"],
    )

    candidate_check: dict[str, Any] | None = None
    candidate_hash: str | None = None
    output_files = [
        paths["crossfit_baseline"],
        paths["crossfit_spatial"],
        paths["transfer_baseline"],
        paths["transfer_spatial"],
    ]
    final_metadata: dict[str, Any] | None = None
    if adopted:
        final_model = ProbabilisticFICRDecision(
            "threshold_meta", config=config
        ).fit(
            augmented_2024,
            actual_2024,
            base_2024[GROUP],
            capacity_kwh=CAPACITY_KWH[GROUP],
        )
        final_g3 = final_model.predict(augmented_2025, base_2025[GROUP])
        stable_path = (
            args.stable_prob_dir
            / "predictions/prob_stable_g1g3_2025_test.parquet"
        )
        stable = pd.read_parquet(stable_path)
        if tuple(stable.columns) != tuple(TARGET_COLS) or not stable.index.equals(
            base_2025.index
        ):
            raise ValueError("stable probabilistic candidate schema/index differs")
        candidate_prediction = stable.copy()
        candidate_prediction[GROUP] = final_g3
        submission = sample.copy()
        for group in TARGET_COLS:
            submission[group] = candidate_prediction[group].to_numpy(dtype=float)
        _atomic_parquet(candidate_prediction, paths["prediction"])
        _atomic_submission(submission, paths["candidate"])
        _atomic_joblib(final_model, paths["model"])
        candidate_check = _verify_submission(paths["candidate"], submission)
        candidate_hash = sha256_file(paths["candidate"])
        final_metadata = final_model.metadata()
        output_files.extend([paths["prediction"], paths["candidate"], paths["model"]])

    results = {
        "warning": (
            "2024 is consumed post-gate development data, not an untouched or "
            "independent gate. No fit-row score was calculated."
        ),
        "variant": {
            "method": "threshold_meta",
            "spatial_feature_columns": list(SPATIAL_FEATURES),
            "spatial_features_only": True,
            "standalone_spatial_replacement": False,
            "core_q07_excluded_reason": (
                "no promoted full-2025 prediction; deployable train/transfer/test "
                "schema is fixed before selection"
            ),
            "config": asdict(config),
        },
        "contracts": {
            "2023_selection": "positive spatial-vs-base-meta delta on Q3 and Q4",
            "2024_confirmation": (
                "fixed 2023-fit recipe must improve spatial-vs-base-meta on H1 and H2"
            ),
            "all_crossfit_indexes_disjoint": True,
            "same_data_fit_score_reported": False,
            "adoption_requires_all_four_periods_positive": True,
            "missing_2024_labels_excluded": missing_2024_labels,
        },
        "crossfit_2023_q3_q4": direction_results,
        "selected_on_2023": selected_on_2023,
        "transfer_2023_to_2024": transfer_results,
        "confirmed_on_2024": confirmed_on_2024,
        "adopted": adopted,
        "candidate_2025": {
            "written": adopted,
            "score_claim": False,
            "path": str(paths["candidate"]) if adopted else None,
            "sha256": candidate_hash,
            "csv_check": candidate_check,
            "final_fit_metadata": final_metadata,
        },
    }
    write_json_atomic(paths["results"], results, overwrite=bool(args.overwrite))
    output_files.append(paths["results"])

    input_files = list(
        dict.fromkeys(
            Path(path).resolve()
            for path in (
                args.raw_dir / "train/train_labels.csv",
                args.spatial_2023,
                args.spatial_2024,
                args.spatial_2025,
                *inputs_2023,
                *inputs_2024,
                *inputs_2025,
                *(
                    [
                        args.stable_prob_dir
                        / "predictions/prob_stable_g1g3_2025_test.parquet"
                    ]
                    if adopted
                    else []
                ),
            )
        )
    )
    manifest = make_manifest(
        artifact_type="baram_probabilistic_spatial_g3_audit",
        parameters={
            "method": "threshold_meta",
            "spatial_features": SPATIAL_FEATURES,
            "config": asdict(config),
            "leaderboard_score_claim": False,
            "untouched_gate_claim": False,
        },
        input_files=input_files,
        output_files=output_files,
        results={
            "selected_on_2023": selected_on_2023,
            "confirmed_on_2024": confirmed_on_2024,
            "adopted": adopted,
            "crossfit_2023_deltas": {
                name: value["delta_vs_baseline_threshold_meta"]
                for name, value in direction_results.items()
            },
            "transfer_2024_deltas": {
                name: value["delta_vs_baseline_threshold_meta"]
                for name, value in transfer_results.items()
            },
            "candidate_2025_sha256": candidate_hash,
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(paths["manifest"], manifest, overwrite=bool(args.overwrite))
    print(
        "2023 spatial-vs-base-meta deltas:",
        {
            name: round(value["delta_vs_baseline_threshold_meta"], 9)
            for name, value in direction_results.items()
        },
    )
    print(
        "2024 fixed-transfer spatial-vs-base-meta deltas:",
        {
            name: round(value["delta_vs_baseline_threshold_meta"], 9)
            for name, value in transfer_results.items()
        },
    )
    print(f"adopted={adopted}; results={paths['results']}")
    if adopted:
        print(f"candidate={paths['candidate']} sha256={candidate_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

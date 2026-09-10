"""Select a deployable g3 threshold-meta diversity feature set without leakage.

Four variants are declared before reading scores: locked/base meta features,
spatial features, pseudo-target features, and their union.  Selection uses only
disjoint 2023 Q3<->Q4 applications.  A non-base variant must improve both
directions and their combined full period; among eligible variants the largest
worst-period delta wins.
Only that fixed winner is transferred to 2024.  A 2025 CSV is produced only if
the fixed transfer also improves both 2024 halves over the base meta model.

2024 is already consumed post-gate development data.  Its results are a
confirmation diagnostic, not an independent gate or leaderboard claim.
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
from scripts.run_probabilistic_spatial_g3 import (  # noqa: E402
    GROUP,
    SPATIAL_2023_COLUMNS,
    SPATIAL_2024_COLUMNS,
    SPATIAL_2025_COLUMNS,
    SPATIAL_FEATURES,
    _spatial_features,
)
from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402
from src.probabilistic import (  # noqa: E402
    ProbabilisticDecisionConfig,
    ProbabilisticFICRDecision,
)


PSEUDO_FEATURES = ("pseudo_l1_cf", "pseudo_q07_cf")
PSEUDO_COLUMNS = {
    "pseudo_l1_cf": "pseudo_l1",
    "pseudo_q07_cf": "pseudo_q07",
}
VARIANT_ORDER = ("base", "spatial", "pseudo", "spatial_pseudo")


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
            "artifacts/experiments/spatial_v2_postgate/spatial_v2_2024_oof.parquet"
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
    pseudo_dir = Path("artifacts/postgate/g3_pseudo_transfer")
    parser.add_argument(
        "--pseudo-2023",
        type=Path,
        default=pseudo_dir / "oof/g3_pseudo_2023h2.parquet",
    )
    parser.add_argument(
        "--pseudo-2024",
        type=Path,
        default=pseudo_dir / "oof/g3_pseudo_2024.parquet",
    )
    parser.add_argument(
        "--pseudo-2025",
        type=Path,
        default=pseudo_dir / "predictions/g3_pseudo_2025.parquet",
    )
    parser.add_argument(
        "--stable-prob-dir",
        type=Path,
        default=Path("artifacts/postgate/probabilistic_ficr"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/probabilistic_diversity_g3"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _pseudo_features(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{path} must have a DatetimeIndex")
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path} index must be unique and sorted")
    if not frame.index.equals(index):
        raise ValueError(f"{path} index differs from exact requested horizon")
    missing = sorted(set(PSEUDO_COLUMNS.values()).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing pseudo columns: {missing}")
    result = frame.loc[:, list(PSEUDO_COLUMNS.values())].rename(
        columns={source: target for target, source in PSEUDO_COLUMNS.items()}
    )
    result = result.loc[:, list(PSEUDO_FEATURES)].astype(float)
    result /= CAPACITY_KWH[GROUP]
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite pseudo features")
    return result


def _variant_frames(
    base: pd.DataFrame,
    spatial: pd.DataFrame,
    pseudo: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    frames = {
        "base": base.copy(),
        "spatial": pd.concat([base, spatial], axis=1),
        "pseudo": pd.concat([base, pseudo], axis=1),
        "spatial_pseudo": pd.concat([base, spatial, pseudo], axis=1),
    }
    if tuple(frames) != VARIANT_ORDER:
        raise AssertionError("variant order changed")
    for name, frame in frames.items():
        if not frame.index.equals(base.index):
            raise ValueError(f"{name} feature index differs")
        if frame.columns.duplicated().any():
            raise ValueError(f"{name} has duplicate feature columns")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} has non-finite features")
    return frames


def _fit(
    features: pd.DataFrame,
    actual: pd.Series,
    base: pd.Series,
    config: ProbabilisticDecisionConfig,
) -> ProbabilisticFICRDecision:
    return ProbabilisticFICRDecision("threshold_meta", config=config).fit(
        features,
        actual,
        base,
        capacity_kwh=CAPACITY_KWH[GROUP],
    )


def _score(actual: pd.Series, prediction: pd.Series) -> dict[str, Any]:
    return _group_summary(actual, prediction, GROUP)


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

    base_features = {
        "2023": _meta_features(candidates_2023, base_2023, GROUP),
        "2024": _meta_features(candidates_2024, base_2024, GROUP),
        "2025": _meta_features(candidates_2025, base_2025, GROUP),
    }
    spatial_features = {
        "2023": _spatial_features(
            args.spatial_2023, base_features["2023"].index, SPATIAL_2023_COLUMNS
        ),
        "2024": _spatial_features(
            args.spatial_2024, base_features["2024"].index, SPATIAL_2024_COLUMNS
        ),
        "2025": _spatial_features(
            args.spatial_2025, base_features["2025"].index, SPATIAL_2025_COLUMNS
        ),
    }
    pseudo_features = {
        "2023": _pseudo_features(args.pseudo_2023, base_features["2023"].index),
        "2024": _pseudo_features(args.pseudo_2024, base_features["2024"].index),
        "2025": _pseudo_features(args.pseudo_2025, base_features["2025"].index),
    }
    variants = {
        year: _variant_frames(
            base_features[year], spatial_features[year], pseudo_features[year]
        )
        for year in ("2023", "2024", "2025")
    }
    for variant in VARIANT_ORDER:
        schema = tuple(variants["2023"][variant].columns)
        if schema != tuple(variants["2024"][variant].columns) or schema != tuple(
            variants["2025"][variant].columns
        ):
            raise ValueError(f"{variant} train/transfer/test feature schema differs")

    q3 = _interval("2023-07-01 01:00:00", "2023-10-01 00:00:00")
    q4 = _interval("2023-10-01 01:00:00", "2024-01-01 00:00:00")
    h1_2024 = _interval("2024-01-01 01:00:00", "2024-07-01 00:00:00")
    h2_2024 = _interval("2024-07-01 01:00:00", "2025-01-01 00:00:00")
    actual_2023 = labels.loc[base_2023.index, GROUP]
    actual_2024 = labels.loc[base_2024.index, GROUP]
    if actual_2023.isna().any():
        raise ValueError("2023 H2 g3 labels are incomplete")
    print(
        "validated variant schemas:",
        {name: variants["2023"][name].shape[1] for name in VARIANT_ORDER},
    )
    if args.dry_run:
        print("dry-run complete: no model fit, score, or output write")
        return 0

    paths: dict[str, Path] = {
        f"crossfit:{variant}": args.out_dir / f"oof/crossfit_2023__{variant}.parquet"
        for variant in VARIANT_ORDER
    }
    paths.update(
        {
            "transfer_base": args.out_dir / "oof/transfer_2023_to_2024__base.parquet",
            "transfer_selected": args.out_dir
            / "oof/transfer_2023_to_2024__selected.parquet",
            "prediction": args.out_dir / "predictions/diversity_g3_2025_test.parquet",
            "candidate": args.out_dir / "prob_stable_g1g3_diversity_2025.csv",
            "model": args.out_dir / "models/diversity_g3_2024_fit.joblib",
            "results": args.out_dir / "results.json",
            "manifest": args.out_dir / "manifest.json",
        }
    )
    _assert_outputs_available(
        [
            *(paths[f"crossfit:{variant}"] for variant in VARIANT_ORDER),
            paths["transfer_base"],
            paths["results"],
            paths["manifest"],
        ],
        bool(args.overwrite),
    )
    config = ProbabilisticDecisionConfig(random_state=args.seed)

    crossfit = {variant: base_2023.copy() for variant in VARIANT_ORDER}
    direction_scores: dict[str, dict[str, Any]] = {}
    direction_deltas = {variant: [] for variant in VARIANT_ORDER}
    for direction, fit_index, apply_index in (
        ("q3_fit_q4_apply", q3, q4),
        ("q4_fit_q3_apply", q4, q3),
    ):
        if len(fit_index.intersection(apply_index)):
            raise AssertionError("2023 fit/application indexes overlap")
        predictions: dict[str, pd.Series] = {}
        metadata: dict[str, Any] = {}
        for variant in VARIANT_ORDER:
            print(f"2023 {direction} {variant}", flush=True)
            model = _fit(
                variants["2023"][variant].loc[fit_index],
                actual_2023.loc[fit_index],
                base_2023.loc[fit_index, GROUP],
                config,
            )
            predictions[variant] = model.predict(
                variants["2023"][variant].loc[apply_index],
                base_2023.loc[apply_index, GROUP],
            )
            crossfit[variant].loc[apply_index, GROUP] = predictions[variant]
            metadata[variant] = model.metadata()
        base_score = _score(actual_2023.loc[apply_index], predictions["base"])
        variant_records: dict[str, Any] = {}
        for variant in VARIANT_ORDER:
            score = _score(actual_2023.loc[apply_index], predictions[variant])
            delta = float(score["score"] - base_score["score"])
            direction_deltas[variant].append(delta)
            variant_records[variant] = {
                "metrics": score,
                "delta_vs_base_meta": delta,
                "fit_metadata": metadata[variant],
            }
        direction_scores[direction] = {
            "fit_rows": len(fit_index),
            "application_rows": len(apply_index),
            "fit_application_overlap": 0,
            "variants": variant_records,
        }

    full_base_score = _score(actual_2023, crossfit["base"][GROUP])
    full_scores: dict[str, Any] = {}
    selection_period_deltas: dict[str, dict[str, float]] = {}
    for variant in VARIANT_ORDER:
        variant_score = _score(actual_2023, crossfit[variant][GROUP])
        full_delta = float(variant_score["score"] - full_base_score["score"])
        full_scores[variant] = {
            "metrics": variant_score,
            "delta_vs_base_meta": full_delta,
        }
        selection_period_deltas[variant] = {
            "full": full_delta,
            "q4_application": float(direction_deltas[variant][0]),
            "q3_application": float(direction_deltas[variant][1]),
        }
    eligible = [
        variant
        for variant in VARIANT_ORDER[1:]
        if all(delta > 0.0 for delta in selection_period_deltas[variant].values())
    ]
    # VARIANT_ORDER is the final deterministic tie-breaker because max keeps
    # the first item for exactly equal keys.
    selected_variant = (
        max(
            eligible,
            key=lambda name: (
                min(selection_period_deltas[name].values()),
                float(np.mean(list(selection_period_deltas[name].values()))),
                -VARIANT_ORDER.index(name),
            ),
        )
        if eligible
        else "base"
    )
    selection = {
        "eligible_both_directions_positive": eligible,
        "criterion": "maximise minimum direction delta; then mean; then fixed order",
        "selected_variant": selected_variant,
        "direction_deltas": direction_deltas,
        "selection_period_deltas": selection_period_deltas,
        "combined_full_scores": full_scores,
    }

    for variant in VARIANT_ORDER:
        _atomic_parquet(crossfit[variant], paths[f"crossfit:{variant}"])

    print(f"selected on 2023: {selected_variant}", flush=True)
    base_transfer_model = _fit(
        variants["2023"]["base"], actual_2023, base_2023[GROUP], config
    )
    base_transfer = base_transfer_model.predict(
        variants["2024"]["base"], base_2024[GROUP]
    )
    _atomic_parquet(
        pd.DataFrame({GROUP: base_transfer}, index=base_2024.index),
        paths["transfer_base"],
    )
    selected_transfer: pd.Series | None = None
    selected_transfer_model: ProbabilisticFICRDecision | None = None
    transfer_audit: dict[str, Any] | None = None
    confirmed = False
    if selected_variant != "base":
        selected_transfer_model = _fit(
            variants["2023"][selected_variant],
            actual_2023,
            base_2023[GROUP],
            config,
        )
        selected_transfer = selected_transfer_model.predict(
            variants["2024"][selected_variant], base_2024[GROUP]
        )
        _atomic_parquet(
            pd.DataFrame({GROUP: selected_transfer}, index=base_2024.index),
            paths["transfer_selected"],
        )
        transfer_audit = {}
        for period, index in (
            ("full", actual_2024.index),
            ("h1", h1_2024),
            ("h2", h2_2024),
        ):
            base_metrics = _score(actual_2024.loc[index], base_transfer.loc[index])
            selected_metrics = _score(
                actual_2024.loc[index], selected_transfer.loc[index]
            )
            transfer_audit[period] = {
                "base_meta": base_metrics,
                "selected_diversity": selected_metrics,
                "delta_vs_base_meta": selected_metrics["score"]
                - base_metrics["score"],
            }
        confirmed = all(
            transfer_audit[period]["delta_vs_base_meta"] > 0.0
            for period in ("full", "h1", "h2")
        )
    adopted = selected_variant != "base" and confirmed

    output_files = [
        *(paths[f"crossfit:{variant}"] for variant in VARIANT_ORDER),
        paths["transfer_base"],
    ]
    if selected_transfer is not None:
        output_files.append(paths["transfer_selected"])
    final_metadata: dict[str, Any] | None = None
    candidate_check: dict[str, Any] | None = None
    candidate_hash: str | None = None
    if adopted:
        final_model = _fit(
            variants["2024"][selected_variant],
            actual_2024,
            base_2024[GROUP],
            config,
        )
        final_g3 = final_model.predict(
            variants["2025"][selected_variant], base_2025[GROUP]
        )
        stable_path = (
            args.stable_prob_dir
            / "predictions/prob_stable_g1g3_2025_test.parquet"
        )
        stable = pd.read_parquet(stable_path)
        if tuple(stable.columns) != tuple(TARGET_COLS) or not stable.index.equals(
            base_2025.index
        ):
            raise ValueError("stable probabilistic candidate schema/index differs")
        prediction = stable.copy()
        prediction[GROUP] = final_g3
        submission = sample.copy()
        for group in TARGET_COLS:
            submission[group] = prediction[group].to_numpy(dtype=float)
        _atomic_parquet(prediction, paths["prediction"])
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
        "predeclared_variants": list(VARIANT_ORDER),
        "feature_sets": {
            "spatial": list(SPATIAL_FEATURES),
            "pseudo": list(PSEUDO_FEATURES),
            "standalone_replacement": False,
        },
        "config": asdict(config),
        "contracts": {
            "selection_data": (
                "2023 disjoint Q3<->Q4 application rows, requiring combined "
                "full plus each application subperiod to improve"
            ),
            "selection_rule": selection["criterion"],
            "transfer_data": "fixed selected variant, 2023 full fit -> 2024 apply",
            "2024_status": "consumed post-gate confirmation only",
            "adoption_rule": (
                "positive fixed-transfer delta on 2024 full, H1, and H2"
            ),
            "same_data_fit_score_reported": False,
            "all_crossfit_indexes_disjoint": True,
            "pseudo_and_spatial_are_features_only": True,
        },
        "crossfit_2023": direction_scores,
        "selection_2023": selection,
        "transfer_2023_to_2024": transfer_audit,
        "confirmed_on_2024": confirmed,
        "adopted": adopted,
        "candidate_2025": {
            "written": adopted,
            "score_claim": False,
            "path": str(paths["candidate"]) if adopted else None,
            "sha256": candidate_hash,
            "csv_check": candidate_check,
            "fit_metadata": final_metadata,
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
                args.pseudo_2023,
                args.pseudo_2024,
                args.pseudo_2025,
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
        artifact_type="baram_probabilistic_diversity_g3_selection",
        parameters={
            "variant_order": VARIANT_ORDER,
            "spatial_features": SPATIAL_FEATURES,
            "pseudo_features": PSEUDO_FEATURES,
            "config": asdict(config),
            "leaderboard_score_claim": False,
            "untouched_gate_claim": False,
        },
        input_files=input_files,
        output_files=output_files,
        results={
            "selection_2023": selection,
            "transfer_2024": transfer_audit,
            "confirmed_on_2024": confirmed,
            "adopted": adopted,
            "candidate_2025_sha256": candidate_hash,
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(paths["manifest"], manifest, overwrite=bool(args.overwrite))
    print("2023 direction deltas:", direction_deltas)
    print("2023 full/Q4/Q3 deltas:", selection_period_deltas)
    print("2024 transfer audit:", None if transfer_audit is None else {
        period: round(value["delta_vs_base_meta"], 9)
        for period, value in transfer_audit.items()
    })
    print(f"adopted={adopted}; results={paths['results']}")
    if adopted:
        print(f"candidate={paths['candidate']} sha256={candidate_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

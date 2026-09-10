"""Execute the preregistered fixed-weight g3 pseudo-candidate replacement.

The exact variants and selection/confirmation rules are read from
``configs/g3_pseudo_ensemble_preregister.json``.  One variant is selected on
2023 H2 (full, Q3, and Q4 all positive versus identity), then that single
variant alone is confirmed on 2024 (full, H1, and H2 all positive).  2024 is
never used to reselect a variant or change a weight, affine term, or clip.
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

from scripts.run_probabilistic_ficr import (  # noqa: E402
    COMMON_CANDIDATES,
    _assert_outputs_available,
    _atomic_parquet,
    _atomic_submission,
    _gate_2024_sources,
    _g3_2023_sources,
    _group_summary,
    _interval,
    _read_labels,
    _test_2025_sources,
    _verify_submission,
)
from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


GROUP = "kpx_group_3"
PSEUDO_COLUMNS = ("pseudo_l1", "pseudo_q07")
PSEUDO_SOURCE_NAMES = {
    "pseudo_l1": "pseudo_l1",
    "pseudo_q07": "pseudo_q07",
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
        "--config",
        type=Path,
        default=Path("configs/g3_pseudo_ensemble_preregister.json"),
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
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/g3_pseudo_candidate_replacement"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "selection_uses_2024",
        "base_group",
        "base_affine_and_clip_unchanged",
        "base_weights_unchanged",
        "variants",
        "selection_rule",
        "confirmation_rule",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"preregister config missing fields: {missing}")
    if config["selection_uses_2024"] is not False:
        raise ValueError("preregister must prohibit 2024 selection")
    if config["base_group"] != GROUP:
        raise ValueError(f"unexpected base_group: {config['base_group']!r}")
    variants = config["variants"]
    if not isinstance(variants, dict) or variants.get("identity") != {}:
        raise ValueError("variants must be an ordered object beginning with identity={}")
    if next(iter(variants)) != "identity":
        raise ValueError("identity must be the first preregistered variant")
    weights = config["base_weights_unchanged"]
    if tuple(weights) != COMMON_CANDIDATES:
        raise ValueError(
            f"base weight names/order differ: {tuple(weights)!r}"
        )
    numeric_weights = np.asarray(list(weights.values()), dtype=float)
    if not np.isfinite(numeric_weights).all() or not np.isclose(
        numeric_weights.sum(), 1.0, atol=1e-12
    ):
        raise ValueError("base weights must be finite and sum exactly to one")
    allowed_replacements = set(PSEUDO_SOURCE_NAMES)
    for name, replacements in variants.items():
        if not isinstance(replacements, dict):
            raise ValueError(f"variant {name!r} must map slots to sources")
        invalid_slots = sorted(set(replacements).difference(weights))
        invalid_sources = sorted(set(replacements.values()).difference(allowed_replacements))
        if invalid_slots or invalid_sources:
            raise ValueError(
                f"invalid {name!r}: slots={invalid_slots}, sources={invalid_sources}"
            )
    affine = config["base_affine_and_clip_unchanged"]
    for field in (
        "scale",
        "bias_kwh",
        "lower_capacity_fraction",
        "upper_capacity_fraction",
    ):
        if field not in affine or not np.isfinite(float(affine[field])):
            raise ValueError(f"invalid affine/clip field {field!r}")
    return config


def _read_pseudo(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{path} must have a DatetimeIndex")
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index):
        raise ValueError(f"{path} index differs from exact expected horizon")
    if tuple(frame.columns) != PSEUDO_COLUMNS:
        raise ValueError(
            f"{path} columns={tuple(frame.columns)!r}, expected={PSEUDO_COLUMNS!r}"
        )
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite predictions")
    return frame.astype(float)


def _assemble_variant(
    candidates: Mapping[str, pd.DataFrame],
    pseudo: pd.DataFrame,
    config: Mapping[str, Any],
    variant: str,
) -> pd.Series:
    weights = config["base_weights_unchanged"]
    replacements = config["variants"][variant]
    index = pseudo.index
    weighted = np.zeros(len(index), dtype=float)
    for slot, weight in weights.items():
        source = replacements.get(slot, slot)
        if source in PSEUDO_SOURCE_NAMES:
            values = pseudo[PSEUDO_SOURCE_NAMES[source]].to_numpy(dtype=float)
        else:
            frame = candidates[source]
            if not frame.index.equals(index):
                raise ValueError(f"candidate {source!r} index differs")
            values = frame[GROUP].to_numpy(dtype=float)
        weighted += float(weight) * values
    affine = config["base_affine_and_clip_unchanged"]
    prediction = float(affine["scale"]) * weighted + float(affine["bias_kwh"])
    prediction = np.clip(
        prediction,
        float(affine["lower_capacity_fraction"]) * CAPACITY_KWH[GROUP],
        float(affine["upper_capacity_fraction"]) * CAPACITY_KWH[GROUP],
    )
    if not np.isfinite(prediction).all():
        raise ValueError(f"variant {variant!r} produced non-finite predictions")
    return pd.Series(prediction, index=index, name=GROUP)


def _period_record(
    actual: pd.Series,
    identity: pd.Series,
    candidate: pd.Series,
) -> dict[str, Any]:
    identity_metrics = _group_summary(actual, identity, GROUP)
    candidate_metrics = _group_summary(actual, candidate, GROUP)
    return {
        "identity": identity_metrics,
        "candidate": candidate_metrics,
        "delta_vs_identity": candidate_metrics["score"] - identity_metrics["score"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_config(args.config)
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
    pseudo_2023 = _read_pseudo(args.pseudo_2023, base_2023.index)
    pseudo_2024 = _read_pseudo(args.pseudo_2024, base_2024.index)
    pseudo_2025 = _read_pseudo(args.pseudo_2025, base_2025.index)

    predictions_2023 = {
        variant: _assemble_variant(candidates_2023, pseudo_2023, config, variant)
        for variant in config["variants"]
    }
    predictions_2024 = {
        variant: _assemble_variant(candidates_2024, pseudo_2024, config, variant)
        for variant in config["variants"]
    }
    predictions_2025 = {
        variant: _assemble_variant(candidates_2025, pseudo_2025, config, variant)
        for variant in config["variants"]
    }
    identity_checks = {
        "2023_max_abs_diff_from_locked_v3": float(
            np.max(np.abs(predictions_2023["identity"] - base_2023[GROUP]))
        ),
        "2024_max_abs_diff_from_locked_v3": float(
            np.max(np.abs(predictions_2024["identity"] - base_2024[GROUP]))
        ),
        "2025_max_abs_diff_from_corrected_v3": float(
            np.max(np.abs(predictions_2025["identity"] - base_2025[GROUP]))
        ),
    }
    if any(value > 1e-8 for value in identity_checks.values()):
        raise AssertionError(f"identity recipe does not reproduce locked v3: {identity_checks}")

    q3 = _interval("2023-07-01 01:00:00", "2023-10-01 00:00:00")
    q4 = _interval("2023-10-01 01:00:00", "2024-01-01 00:00:00")
    h1_2024 = _interval("2024-01-01 01:00:00", "2024-07-01 00:00:00")
    h2_2024 = _interval("2024-07-01 01:00:00", "2025-01-01 00:00:00")
    actual_2023 = labels.loc[base_2023.index, GROUP]
    actual_2024 = labels.loc[base_2024.index, GROUP]
    if actual_2023.isna().any():
        raise ValueError("2023 H2 labels must be complete")
    print(f"identity reproduction: {identity_checks}")
    if args.dry_run:
        print("dry-run complete: schemas/formula validated; no score/output write")
        return 0

    paths: dict[str, Path] = {
        f"2023:{variant}": args.out_dir / f"oof/2023h2__{variant}.parquet"
        for variant in config["variants"]
    }
    paths.update(
        {
            "2024_identity": args.out_dir / "oof/2024__identity.parquet",
            "2024_selected": args.out_dir / "oof/2024__selected.parquet",
            "prediction": args.out_dir / "predictions/g3_pseudo_replacement_2025.parquet",
            "candidate": args.out_dir / "g3_pseudo_replacement_2025.csv",
            "results": args.out_dir / "results.json",
            "manifest": args.out_dir / "manifest.json",
        }
    )
    _assert_outputs_available(
        [
            *(paths[f"2023:{variant}"] for variant in config["variants"]),
            paths["2024_identity"],
            paths["results"],
            paths["manifest"],
        ],
        bool(args.overwrite),
    )

    selection_records: dict[str, Any] = {}
    eligible: list[str] = []
    minimum_deltas: dict[str, float] = {}
    for variant in config["variants"]:
        records = {
            period: _period_record(
                actual_2023.loc[index],
                predictions_2023["identity"].loc[index],
                predictions_2023[variant].loc[index],
            )
            for period, index in (
                ("full", actual_2023.index),
                ("q3", q3),
                ("q4", q4),
            )
        }
        deltas = [record["delta_vs_identity"] for record in records.values()]
        minimum_deltas[variant] = float(min(deltas))
        qualifies = variant != "identity" and all(delta > 0.0 for delta in deltas)
        if qualifies:
            eligible.append(variant)
        selection_records[variant] = {
            "periods": records,
            "minimum_delta": minimum_deltas[variant],
            "eligible": qualifies,
        }
        _atomic_parquet(
            pd.DataFrame({GROUP: predictions_2023[variant]}),
            paths[f"2023:{variant}"],
        )
    # sorted() plus max()'s first-maximum behaviour implements the exact
    # lexicographic tie break without introducing a secondary score criterion.
    selected = (
        max(sorted(eligible), key=lambda name: minimum_deltas[name])
        if eligible
        else "identity"
    )

    _atomic_parquet(
        pd.DataFrame({GROUP: predictions_2024["identity"]}), paths["2024_identity"]
    )
    confirmation_records: dict[str, Any] | None = None
    confirmed = False
    if selected != "identity":
        confirmation_records = {
            period: _period_record(
                actual_2024.loc[index],
                predictions_2024["identity"].loc[index],
                predictions_2024[selected].loc[index],
            )
            for period, index in (
                ("full", actual_2024.index),
                ("h1", h1_2024),
                ("h2", h2_2024),
            )
        }
        confirmed = all(
            record["delta_vs_identity"] > 0.0
            for record in confirmation_records.values()
        )
        _atomic_parquet(
            pd.DataFrame({GROUP: predictions_2024[selected]}),
            paths["2024_selected"],
        )

    output_files = [
        *(paths[f"2023:{variant}"] for variant in config["variants"]),
        paths["2024_identity"],
    ]
    if selected != "identity":
        output_files.append(paths["2024_selected"])
    candidate_check: dict[str, Any] | None = None
    candidate_hash: str | None = None
    if confirmed:
        prediction = base_2025.copy()
        prediction[GROUP] = predictions_2025[selected]
        submission = sample.copy()
        for group in TARGET_COLS:
            submission[group] = prediction[group].to_numpy(dtype=float)
        _atomic_parquet(prediction, paths["prediction"])
        _atomic_submission(submission, paths["candidate"])
        candidate_check = _verify_submission(paths["candidate"], submission)
        candidate_hash = sha256_file(paths["candidate"])
        output_files.extend([paths["prediction"], paths["candidate"]])

    results = {
        "warning": (
            "2024 is consumed post-gate development data, not an untouched or "
            "independent gate. No leaderboard score is claimed."
        ),
        "preregister_config": str(args.config),
        "preregister_config_sha256": sha256_file(args.config),
        "identity_reproduction": identity_checks,
        "selection_2023": {
            "records": selection_records,
            "eligible": eligible,
            "minimum_deltas": minimum_deltas,
            "selected_variant": selected,
            "rule": config["selection_rule"],
        },
        "confirmation_2024": {
            "evaluated_variant": selected if selected != "identity" else None,
            "records": confirmation_records,
            "confirmed": confirmed,
            "rule": config["confirmation_rule"],
            "reselection_performed": False,
        },
        "candidate_2025": {
            "written": confirmed,
            "score_claim": False,
            "path": str(paths["candidate"]) if confirmed else None,
            "sha256": candidate_hash,
            "csv_check": candidate_check,
        },
    }
    write_json_atomic(paths["results"], results, overwrite=bool(args.overwrite))
    output_files.append(paths["results"])
    inputs = list(
        dict.fromkeys(
            Path(path).resolve()
            for path in (
                args.config,
                args.raw_dir / "train/train_labels.csv",
                args.pseudo_2023,
                args.pseudo_2024,
                args.pseudo_2025,
                *inputs_2023,
                *inputs_2024,
                *inputs_2025,
            )
        )
    )
    manifest = make_manifest(
        artifact_type="baram_g3_pseudo_fixed_candidate_replacement",
        parameters={
            "preregister_sha256": sha256_file(args.config),
            "selection_uses_2024": False,
            "weights_affine_clip_changed": False,
            "leaderboard_score_claim": False,
            "untouched_gate_claim": False,
        },
        input_files=inputs,
        output_files=output_files,
        results={
            "identity_reproduction": identity_checks,
            "eligible_2023": eligible,
            "selected_2023": selected,
            "confirmation_2024": confirmation_records,
            "confirmed": confirmed,
            "candidate_2025_sha256": candidate_hash,
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(paths["manifest"], manifest, overwrite=bool(args.overwrite))
    print("2023 minimum deltas:", minimum_deltas)
    print(f"selected_2023={selected}")
    print(
        "2024 confirmation:",
        None
        if confirmation_records is None
        else {
            period: round(record["delta_vs_identity"], 9)
            for period, record in confirmation_records.items()
        },
    )
    print(f"confirmed={confirmed}; results={paths['results']}")
    if confirmed:
        print(f"candidate={paths['candidate']} sha256={candidate_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

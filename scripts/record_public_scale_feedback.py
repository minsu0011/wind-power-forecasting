"""Record and audit already-observed global scale feedback without extrapolation.

This is a post-feedback forensic recorder, not a model-selection runner.  It
refuses to overwrite its output directory and never creates a submission CSV.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence
import uuid

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/public_scale_feedback_20260808_1630_preregister.json"
)
DEFAULT_QUADRATIC_ADDENDUM = PROJECT_ROOT / (
    "configs/public_scale_feedback_20260808_1630_quadratic_addendum_preregister.json"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "artifacts/postgate/public_scale_feedback_20260808_1630"
)
TARGET_COLS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
SUBMISSION_COLS = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
CAPACITIES = {"kpx_group_1": 21_600.0, "kpx_group_2": 21_600.0, "kpx_group_3": 21_000.0}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--quadratic-addendum", type=Path, default=DEFAULT_QUADRATIC_ADDENDUM
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def staged_output_record(staging_path: Path, final_path: Path) -> dict[str, Any]:
    """Hash a staged file while manifesting its immutable post-rename path."""

    value = record(staging_path)
    value["path"] = str(final_path.resolve())
    return value


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: JSON root must be an object")
    return value


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def metric_identity(entry: dict[str, Any], tolerance: float) -> float:
    score = float(entry["score"])
    one_minus_nmae = float(entry["one_minus_nmae"])
    ficr = float(entry["ficr"])
    values = np.asarray([score, one_minus_nmae, ficr], dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{entry['label']}: metric components must be finite and in [0,1]")
    residual = score - 0.5 * (one_minus_nmae + ficr)
    if abs(residual) > tolerance:
        raise ValueError(f"{entry['label']}: score identity residual {residual} exceeds {tolerance}")
    return float(residual)


def read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(frame.columns) != SUBMISSION_COLS:
        raise ValueError(f"{path}: unexpected schema {tuple(frame.columns)}")
    if len(frame) != 8760:
        raise ValueError(f"{path}: expected 8760 rows; got {len(frame)}")
    values = frame.loc[:, TARGET_COLS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite prediction")
    return frame


def delta(later: dict[str, Any], earlier: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(later[key]) - float(earlier[key])
        for key in ("score", "one_minus_nmae", "ficr")
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    addendum_path = args.quadratic_addendum.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output directory: {out_dir}")
    config = read_object(config_path)
    addendum = read_object(addendum_path)
    required_false = (
        "private_champion",
        "strict_validation_claim",
    )
    if not config.get("registered_after_feedback_was_observed"):
        raise ValueError("config must disclose registration after feedback")
    if not config.get("post_feedback_descriptive_only"):
        raise ValueError("config must be descriptive-only")
    if not config.get("public_adaptive") or not config.get("selection_unsafe"):
        raise ValueError("config must mark Public adaptation and selection unsafety")
    if any(config.get(key) is not False for key in required_false):
        raise ValueError("config cannot claim strict validation or Private champion status")
    if not config.get("submission_csv_creation_forbidden"):
        raise ValueError("config must forbid submission CSV creation")
    if not addendum.get("registered_after_feedback_was_observed"):
        raise ValueError("quadratic addendum must disclose post-feedback registration")
    if not addendum.get("post_feedback_descriptive_only"):
        raise ValueError("quadratic addendum must be descriptive-only")
    if addendum.get("frozen_descriptive_quadratic", {}).get("recommendation_allowed") is not False:
        raise ValueError("quadratic interpolation cannot be recommendation-eligible")
    if addendum.get("frozen_descriptive_quadratic", {}).get(
        "statistical_confidence_interval_estimable"
    ) is not False:
        raise ValueError("three-point interpolation cannot claim an estimable confidence interval")

    entries = list(config["reported_public_inputs"])
    by_label = {str(entry["label"]): entry for entry in entries}
    expected_labels = {"base_v4", "global_scale_092", "global_scale_095"}
    if set(by_label) != expected_labels or len(entries) != len(expected_labels):
        raise ValueError("reported_public_inputs must contain each frozen label exactly once")
    tolerance = float(config["frozen_analysis"]["metric_identity_tolerance"])

    frames: dict[str, pd.DataFrame] = {}
    file_audits: dict[str, Any] = {}
    identities: dict[str, float] = {}
    input_paths: list[Path] = [config_path, addendum_path]
    for label in ("base_v4", "global_scale_092", "global_scale_095"):
        entry = by_label[label]
        path = resolve_project_path(str(entry["file"]))
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{label}: file SHA differs from preregistration")
        frame = read_submission(path)
        frames[label] = frame
        input_paths.append(path)
        identities[label] = metric_identity(entry, tolerance)
        file_audits[label] = {
            **record(path),
            "utf8_sig_bom": path.read_bytes()[:3] == b"\xef\xbb\xbf",
            "rows": int(len(frame)),
            "columns_exact": tuple(frame.columns) == SUBMISSION_COLS,
            "finite": bool(np.isfinite(frame.loc[:, TARGET_COLS].to_numpy(float)).all()),
        }

    base = frames["base_v4"]
    base_values = base.loc[:, TARGET_COLS].to_numpy(float)
    normalized_max = max(
        float(base[group].max()) / CAPACITIES[group] for group in TARGET_COLS
    )
    for label in ("global_scale_092", "global_scale_095"):
        frame = frames[label]
        if not frame.loc[:, ["forecast_id", "forecast_kst_dtm"]].equals(
            base.loc[:, ["forecast_id", "forecast_kst_dtm"]]
        ):
            raise ValueError(f"{label}: identifiers or timestamps differ from base")
        scale = float(by_label[label]["scale"])
        max_difference = float(
            np.max(
                np.abs(
                    frame.loc[:, TARGET_COLS].to_numpy(float) - scale * base_values
                )
            )
        )
        if max_difference > 5.1e-7:
            raise ValueError(f"{label}: scaled formula difference {max_difference} is too large")
        file_audits[label]["ids_and_timestamps_exact_vs_base"] = True
        file_audits[label]["max_abs_scale_formula_csv_rounding_difference"] = max_difference

    ordered = sorted(entries, key=lambda entry: float(entry["scale"]))
    intervals: list[dict[str, Any]] = []
    for lower, upper in zip(ordered, ordered[1:]):
        a = float(lower["scale"])
        b = float(upper["scale"])
        width = b - a
        loss_a = 1.0 - float(lower["one_minus_nmae"])
        loss_b = 1.0 - float(upper["one_minus_nmae"])
        nmae_secant = (loss_b - loss_a) / width
        ficr_delta = float(upper["ficr"]) - float(lower["ficr"])
        intervals.append(
            {
                "lower_label": lower["label"],
                "upper_label": upper["label"],
                "lower_scale": a,
                "upper_scale": b,
                "scale_width": width,
                "absolute_loss_secant_per_unit_scale": float(nmae_secant),
                "interpretation": (
                    "macro normalized signed forecast-mass secant; positive means "
                    "raising forecasts worsened NMAE over this interval"
                ),
                "conservative_minimum_macro_evaluated_row_fraction_with_positive_loss_contribution": float(
                    max(nmae_secant, 0.0) / normalized_max
                ),
                "ficr_delta_when_raising_scale": ficr_delta,
                "ficr_secant_per_unit_scale": float(ficr_delta / width),
                "conservative_minimum_macro_normalized_actual_energy_mass_whose_settlement_class_changed": abs(
                    ficr_delta
                ),
                "settlement_bound_note": (
                    "Uses the maximum possible per-energy FICR contribution change of 1; "
                    "cancellation can make the true changed mass larger."
                ),
            }
        )
    low = ordered[0]
    high = ordered[-1]
    width = float(high["scale"]) - float(low["scale"])
    total_nmae_secant = (
        (1.0 - float(high["one_minus_nmae"]))
        - (1.0 - float(low["one_minus_nmae"]))
    ) / width
    total_ficr_delta = float(high["ficr"]) - float(low["ficr"])
    intervals.append(
        {
            "lower_label": low["label"],
            "upper_label": high["label"],
            "lower_scale": float(low["scale"]),
            "upper_scale": float(high["scale"]),
            "scale_width": width,
            "absolute_loss_secant_per_unit_scale": float(total_nmae_secant),
            "interpretation": (
                "macro normalized signed forecast-mass secant; positive means "
                "raising forecasts worsened NMAE over this interval"
            ),
            "conservative_minimum_macro_evaluated_row_fraction_with_positive_loss_contribution": float(
                max(total_nmae_secant, 0.0) / normalized_max
            ),
            "ficr_delta_when_raising_scale": total_ficr_delta,
            "ficr_secant_per_unit_scale": float(total_ficr_delta / width),
            "conservative_minimum_macro_normalized_actual_energy_mass_whose_settlement_class_changed": abs(
                total_ficr_delta
            ),
            "settlement_bound_note": (
                "Uses the maximum possible per-energy FICR contribution change of 1; "
                "cancellation can make the true changed mass larger."
            ),
        }
    )

    results_path = PROJECT_ROOT / "artifacts/postgate/public_scale_probe/results.json"
    results = read_object(results_path)
    input_paths.append(results_path)
    proxy_comparison: dict[str, Any] = {}
    proxy_labels = {
        "global_scale_092": "scale_092",
        "global_scale_095": "scale_095",
    }
    for label, proxy_label in proxy_labels.items():
        actual = by_label[label]
        proxy = results["probes"][proxy_label]["public_bias_analogy_only"]
        proxy_comparison[label] = {
            "actual_minus_analogy": {
                key: float(actual[key]) - float(proxy[key])
                for key in ("score", "one_minus_nmae", "ficr")
            },
            "analogy": {
                key: float(proxy[key])
                for key in ("score", "one_minus_nmae", "ficr")
            },
            "actual": {
                key: float(actual[key])
                for key in ("score", "one_minus_nmae", "ficr")
            },
            "analogy_declared_not_a_leaderboard_prediction": bool(
                proxy["not_a_leaderboard_prediction"]
            ),
        }

    comparisons = {
        "global_scale_092_minus_base_v4": delta(
            by_label["global_scale_092"], by_label["base_v4"]
        ),
        "global_scale_095_minus_base_v4": delta(
            by_label["global_scale_095"], by_label["base_v4"]
        ),
        "global_scale_095_minus_global_scale_092": delta(
            by_label["global_scale_095"], by_label["global_scale_092"]
        ),
    }
    actual_order = sorted(entries, key=lambda entry: float(entry["score"]), reverse=True)
    proxy_order = sorted(
        ("global_scale_092", "global_scale_095"),
        key=lambda label: proxy_comparison[label]["analogy"]["score"],
        reverse=True,
    )

    quadratic_order = list(addendum["frozen_descriptive_quadratic"]["input_order"])
    if quadratic_order != ["global_scale_092", "global_scale_095", "base_v4"]:
        raise ValueError("unexpected frozen quadratic input order")
    quadratic_scales = np.asarray(
        [float(by_label[label]["scale"]) for label in quadratic_order], dtype=float
    )

    def fit_quadratic(component: str) -> dict[str, float]:
        values = np.asarray(
            [float(by_label[label][component]) for label in quadratic_order], dtype=float
        )
        a, b, c = np.polyfit(quadratic_scales, values, 2)
        vertex = float(-b / (2.0 * a)) if a < 0.0 else float("nan")
        vertex_value = float(np.polyval([a, b, c], vertex)) if np.isfinite(vertex) else float("nan")
        return {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "vertex_scale": vertex,
            "vertex_value": vertex_value,
        }

    score_quadratic = fit_quadratic("score")
    nmae_quadratic = fit_quadratic("one_minus_nmae")
    ficr_quadratic = fit_quadratic("ficr")
    score_advantage_095_vs_base = float(
        by_label["global_scale_095"]["score"] - by_label["base_v4"]["score"]
    )
    if not np.isclose(score_advantage_095_vs_base, 0.0006668153, rtol=0.0, atol=5e-14):
        raise ValueError("unexpected exact 0.95 score advantage")
    stress_vertices: list[float] = []
    base_scores = np.asarray(
        [float(by_label[label]["score"]) for label in quadratic_order], dtype=float
    )
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        stressed = base_scores + score_advantage_095_vs_base * np.asarray(signs)
        a, b, _ = np.polyfit(quadratic_scales, stressed, 2)
        if a < 0.0:
            stress_vertices.append(float(-b / (2.0 * a)))
    if not stress_vertices:
        raise ValueError("quadratic stress produced no finite concave vertex")
    lower_ficr_secant = abs(
        float(by_label["global_scale_095"]["ficr"])
        - float(by_label["global_scale_092"]["ficr"])
    ) / 0.03
    upper_ficr_secant = abs(
        float(by_label["base_v4"]["ficr"])
        - float(by_label["global_scale_095"]["ficr"])
    ) / 0.05
    daily = dict(addendum["daily_submission_state_after_feedback"])
    if int(daily["used"]) != 5 or int(daily["limit"]) != 5 or int(daily["remaining"]) != 0:
        raise ValueError("daily submission state must record 5/5 used and zero remaining")

    analysis = {
        "schema_version": 1,
        "artifact_type": "baram_public_scale_feedback_forensics",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "post_feedback_descriptive_only": True,
        "public_adaptive": True,
        "selection_unsafe": True,
        "private_champion": False,
        "strict_validation_claim": False,
        "submission_csv_created": False,
        "file_audits": file_audits,
        "metric_identity_residuals": identities,
        "component_deltas": comparisons,
        "aggregate_constraints": {
            "base_prediction_max_capacity_fraction": normalized_max,
            "scale_intervals": intervals,
            "nmae_convexity_conclusion": (
                "The positive [0.92,0.95] absolute-loss secant proves only that any "
                "global-scalar NMAE minimizer is below 0.95; it does not locate it."
            ),
            "ficr_conclusion": (
                "Raising scale improved net FICR on both observed intervals, but aggregate "
                "net changes do not reveal which rows, regimes, or groups crossed 6%/8% thresholds."
            ),
        },
        "proxy_refutation": {
            "by_candidate": proxy_comparison,
            "analogy_candidate_order": proxy_order,
            "actual_candidate_order": [
                entry["label"] for entry in actual_order if entry["label"] != "base_v4"
            ],
            "candidate_order_reversed": proxy_order
            != [entry["label"] for entry in actual_order if entry["label"] != "base_v4"],
            "conclusion": (
                "The consumed 2024 OOF curve analogy failed as a transferable candidate-ranking "
                "device and was already explicitly marked not a leaderboard prediction."
            ),
        },
        "descriptive_quadratic_only": {
            "score_fit": score_quadratic,
            "component_fits": {
                "one_minus_nmae": nmae_quadratic,
                "ficr": ficr_quadratic,
            },
            "interpolated_vertex_scale": score_quadratic["vertex_scale"],
            "interpolated_vertex_score": score_quadratic["vertex_value"],
            "interpolated_uplift_vs_observed_095": float(
                score_quadratic["vertex_value"]
                - float(by_label["global_scale_095"]["score"])
            ),
            "interpolated_uplift_vs_base": float(
                score_quadratic["vertex_value"] - float(by_label["base_v4"]["score"])
            ),
            "observed_095_score_advantage_vs_base_exact": score_advantage_095_vs_base,
            "residual_degrees_of_freedom": 0,
            "statistical_confidence_interval_estimable": False,
            "recommendation_allowed": False,
            "risk_quantification": {
                "adjacent_ficr_secant_ratio": float(
                    lower_ficr_secant / upper_ficr_secant
                ),
                "one_minus_nmae_vs_ficr_component_vertex_gap": abs(
                    nmae_quadratic["vertex_scale"] - ficr_quadratic["vertex_scale"]
                ),
                "stress_epsilon_equal_observed_095_advantage": score_advantage_095_vs_base,
                "independent_plus_minus_epsilon_vertex_min": min(stress_vertices),
                "independent_plus_minus_epsilon_vertex_max": max(stress_vertices),
                "stress_is_not_confidence_interval": True,
            },
            "warning": (
                "This unique three-point interpolation has zero residual degrees of freedom. "
                "The official FICR response is discontinuous at 6% and 8% errors, so the "
                "vertex is neither an uncertainty-calibrated estimate nor a submission recommendation."
            ),
        },
        "daily_submission_state": daily,
        "identifiability": {
            "global_fixed_scalar": (
                "0.95 is the best of the three observed scalar settings by Public total score, "
                "but no unobserved scalar or nonlinear response is validated by three threshold-discontinuous points."
            ),
            "quadratic_or_other_nonlinear_fit": (
                "Three points can interpolate a three-parameter curve with zero residual degrees "
                "of freedom; FICR discontinuities invalidate smooth-shape identification."
            ),
            "group_correction": (
                "Each aggregate component delta is only the mean of three unknown group deltas. "
                "No group-only intervention was observed, so group effects are non-identifiable."
            ),
            "regime_correction": (
                "No regime-only intervention or regime-level Public metric was observed; any "
                "regime rule selected here would be unconstrained leaderboard overfit."
            ),
            "justified_public_based_new_correction": False,
        },
        "next_model_implications": [
            "Treat global overprediction as an NMAE signal, not as permission to extrapolate another Public scalar.",
            "On untouched rolling OOF only, estimate conditional residual distributions and settlement-margin probabilities for the 6% and 8% capacity error bands.",
            "Cross-fit a low-complexity action gate that trims forecasts when expected NMAE benefit dominates while preserving or raising forecasts on FICR-sensitive high-energy rows.",
            "Require year/season/group slice stability and compare against the unmodified base before creating any future submission artifact.",
        ],
        "decision": {
            "promote_submission": False,
            "create_submission": False,
            "scalar_extrapolation_allowed": False,
            "group_or_regime_public_correction_allowed": False,
        },
    }

    staging = out_dir.with_name(f"{out_dir.name}.staging-{uuid.uuid4().hex}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        shutil.copyfile(config_path, staging / "preregister.json")
        shutil.copyfile(addendum_path, staging / "quadratic_addendum_preregister.json")
        feedback = {
            "schema_version": 1,
            "source": "user-reported DACON Public results in the active conversation",
            "received_after_both_submissions": True,
            "submissions": entries,
            "warning": "Public feedback is not strict validation or a Private-score guarantee.",
        }
        atomic_json(staging / "feedback_input.json", feedback)
        atomic_json(staging / "analysis.json", analysis)
        input_paths.append(Path(__file__).resolve())
        manifest = {
            "schema_version": 1,
            "artifact_type": "baram_public_scale_feedback_forensics_manifest",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "immutable_no_overwrite": True,
            "post_feedback_descriptive_only": True,
            "public_adaptive": True,
            "selection_unsafe": True,
            "private_champion": False,
            "strict_validation_claim": False,
            "submission_csv_created": False,
            "inputs": [record(path) for path in input_paths],
            "outputs": [
                staged_output_record(
                    staging / "preregister.json", out_dir / "preregister.json"
                ),
                staged_output_record(
                    staging / "quadratic_addendum_preregister.json",
                    out_dir / "quadratic_addendum_preregister.json",
                ),
                staged_output_record(
                    staging / "feedback_input.json", out_dir / "feedback_input.json"
                ),
                staged_output_record(
                    staging / "analysis.json", out_dir / "analysis.json"
                ),
            ],
        }
        atomic_json(staging / "manifest.json", manifest)
        if out_dir.exists():
            raise FileExistsError(f"refusing race-time overwrite: {out_dir}")
        staging.rename(out_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "manifest_sha256": sha256_file(out_dir / "manifest.json"),
                "submission_csv_created": False,
                "justified_public_based_new_correction": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

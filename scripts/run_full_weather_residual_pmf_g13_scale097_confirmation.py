"""Evaluate the one frozen G1/G3 residual-PMF transfer on scale-0.97 recent-v4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_catboost_multiquantile_bayes as bounded  # noqa: E402
from scripts import run_ficr_bayes_decision_strict as strict  # noqa: E402
from src.manifest import describe_file, sha256_file, utc_now  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402


CONFIG = PROJECT_DIR / "configs/full_weather_residual_pmf_g13_scale097_rescue_preregister_v2.json"
CONFIG_SHA256 = "7f50ff26d32b6e6c2c23349f262f2a19df511603f5e2b0df3be3691155824bba"
RECENT = PROJECT_DIR / "artifacts/oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet"
RECENT_SHA256 = "47d09dfca7119bf8d1298a609ee47fc8280bd9c0acfac5966f98d4d3e52b04e9"
DELTA = PROJECT_DIR / "artifacts/postgate/full_weather_residual_pmf_v1/stage2/primary_delta_cf_2024.parquet"
DELTA_SHA256 = "6e88e46273c7447ad4bee9d62a49252d036408a49faa0f487a75dd4031516e82"
LABELS = Path(r"data/local/open/train/train_labels.csv")
LABELS_SHA256 = "47bb64252195cf4734e67394d6e50485f27a608def3b5a8791fcc7674bbceb03"
OUT_DIR = PROJECT_DIR / "artifacts/postgate/full_weather_residual_pmf_g13_scale097_rescue_v2"
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise AssertionError(f"frozen input changed: {path}")


def _comparison(actual: pd.DataFrame, base: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    segments = bounded._year_segments(2024)
    result: dict[str, object] = {"mixed": {}, "groups": {}}
    for segment in SEGMENTS:
        idx = segments[segment]
        before = score_details(actual.loc[idx, list(TARGET_COLS)], base.loc[idx, list(TARGET_COLS)]).as_dict()
        after = score_details(actual.loc[idx, list(TARGET_COLS)], candidate.loc[idx, list(TARGET_COLS)]).as_dict()
        result["mixed"][segment] = {
            "baseline": before,
            "candidate": after,
            "delta_total": float(after["total_score"] - before["total_score"]),
            "delta_one_minus_nmae": float(after["one_minus_nmae"] - before["one_minus_nmae"]),
            "delta_ficr": float(after["ficr"] - before["ficr"]),
        }
    for group in TARGET_COLS:
        group_result: dict[str, object] = {}
        for segment in SEGMENTS:
            idx = segments[segment]
            before = score_details(
                actual.loc[idx, [group]], base.loc[idx, [group]], target_cols=(group,)
            ).as_dict()
            after = score_details(
                actual.loc[idx, [group]], candidate.loc[idx, [group]], target_cols=(group,)
            ).as_dict()
            group_result[segment] = {
                "delta_total": float(after["total_score"] - before["total_score"]),
                "delta_one_minus_nmae": float(after["one_minus_nmae"] - before["one_minus_nmae"]),
                "delta_ficr": float(after["ficr"] - before["ficr"]),
            }
        result["groups"][group] = group_result
    return result


def main() -> int:
    if OUT_DIR.exists():
        raise FileExistsError(f"fresh output required: {OUT_DIR}")
    for path, digest in ((CONFIG, CONFIG_SHA256), (RECENT, RECENT_SHA256), (DELTA, DELTA_SHA256)):
        _verify(path, digest)

    index = strict._year_index(2024)
    recent = strict._read_prediction(RECENT, index, required_columns=TARGET_COLS)
    delta = strict._read_prediction(DELTA, index, required_columns=TARGET_COLS)
    base = recent.copy()
    candidate = recent.copy()
    for group in TARGET_COLS:
        cap = CAPACITY_KWH[group]
        base_cf = np.clip(recent[group].to_numpy(dtype=np.float64) / cap * 0.97, 0.0, 1.02)
        base[group] = base_cf * cap
        if group == "kpx_group_2":
            candidate[group] = base[group].to_numpy(dtype=np.float64)
        else:
            candidate[group] = np.clip(base_cf + delta[group].to_numpy(dtype=np.float64), 0.0, 1.02) * cap
    if not np.array_equal(base["kpx_group_2"].to_numpy(), candidate["kpx_group_2"].to_numpy()):
        raise AssertionError("G2 identity changed")

    strict._atomic_parquet(base, OUT_DIR / "confirmation/base_scale097_2024.parquet")
    strict._atomic_parquet(candidate, OUT_DIR / "confirmation/candidate_g13_2024.parquet")
    lock = OUT_DIR / "confirmation_candidate_before_label_lock.json"
    strict._write_json(lock, {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister": describe_file(CONFIG),
        "base": describe_file(OUT_DIR / "confirmation/base_scale097_2024.parquet"),
        "candidate": describe_file(OUT_DIR / "confirmation/candidate_g13_2024.parquet"),
        "base_array_sha256": _array_sha256(base.to_numpy(dtype=np.float64)),
        "candidate_array_sha256": _array_sha256(candidate.to_numpy(dtype=np.float64)),
        "g2_bit_identity": True,
        "application_label_values_read_before_lock": 0,
        "public_feedback_or_subset_values_read": 0,
    })

    _verify(LABELS, LABELS_SHA256)
    labels = strict._read_full_labels(LABELS).loc[index, list(TARGET_COLS)]
    comparisons = _comparison(labels, base, candidate)
    mixed = comparisons["mixed"]
    positive = sum(float(mixed[s]["delta_total"]) > 0.0 for s in SEGMENTS)
    checks = {
        "mixed_full_total_ge_0p00025": float(mixed["full"]["delta_total"]) >= 0.00025,
        "mixed_h2_total_positive": float(mixed["H2"]["delta_total"]) > 0.0,
        "mixed_full_ficr_positive": float(mixed["full"]["delta_ficr"]) > 0.0,
        "mixed_h2_ficr_positive": float(mixed["H2"]["delta_ficr"]) > 0.0,
        "mixed_positive_segments_ge_5": positive >= 5,
        "g1_full_positive": float(comparisons["groups"]["kpx_group_1"]["full"]["delta_total"]) > 0.0,
        "g3_full_positive": float(comparisons["groups"]["kpx_group_3"]["full"]["delta_total"]) > 0.0,
    }
    passed = all(checks.values())
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister": describe_file(CONFIG),
        "candidate_before_label_lock": describe_file(lock),
        "selection_unsafe_posthoc": True,
        "comparisons": comparisons,
        "positive_mixed_segments": positive,
        "gate_checks": checks,
        "promoted": passed,
        "conditional_final_fit_allowed": passed,
        "public_feedback_used": False,
        "year_2025_values_read": 0,
        "csv_created": False,
    }
    strict._write_json(OUT_DIR / "confirmation_results.json", result)
    strict._write_json(OUT_DIR / "promotion_lock.json", {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister": describe_file(CONFIG),
        "confirmation_results": describe_file(OUT_DIR / "confirmation_results.json"),
        "promoted": passed,
        "final_fit_allowed": passed,
        "no_rescue_or_retune": True,
    })
    print(json.dumps({"promoted": passed, "positive": positive, "checks": checks, "mixed": {s: mixed[s]["delta_total"] for s in SEGMENTS}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

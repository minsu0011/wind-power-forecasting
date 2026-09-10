"""Independent formula, metric, weight, and provenance audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_DIR / "artifacts/postgate/density_ratio_reweight_strict_v1"
PREREG_SHA = "cec7ed311ad3b9358a7c5c97b66f788938de5fe17a77d27417e0a5b3f96c794e"
FEATURE_SHA = "990148af97c89aad5b21c12388ffbfeb214819424e14c8950ed0f37c94e60feb"
GROUPS = ("kpx_group_1", "kpx_group_2")
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independent_metric(actual: np.ndarray, prediction: np.ndarray, capacity: float) -> dict[str, float]:
    valid = np.isfinite(actual) & (actual >= 0.10 * capacity)
    actual = actual[valid]
    prediction = prediction[valid]
    error = np.abs(prediction - actual) / capacity
    payment = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    one_minus_nmae = float(1.0 - error.mean())
    ficr = float(np.sum(actual * payment) / np.sum(actual * 4.0))
    return {
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
        "total": 0.5 * (one_minus_nmae + ficr),
    }


def segment_masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    year = int(index.min().year)
    b1 = pd.Timestamp(year, 4, 1, 0)
    b2 = pd.Timestamp(year, 7, 1, 0)
    b3 = pd.Timestamp(year, 10, 1, 0)
    end = pd.Timestamp(year + 1, 1, 1, 0)
    return {
        "full": np.ones(len(index), dtype=bool),
        "H1": np.asarray(index <= b2),
        "H2": np.asarray((index > b2) & (index <= end)),
        "Q1": np.asarray(index <= b1),
        "Q2": np.asarray((index > b1) & (index <= b2)),
        "Q3": np.asarray((index > b2) & (index <= b3)),
        "Q4": np.asarray((index > b3) & (index <= end)),
    }


def main() -> None:
    prereg = OUT_DIR / "preregister.json"
    features = OUT_DIR / "domain_features.txt"
    if sha256_file(prereg) != PREREG_SHA or sha256_file(features) != FEATURE_SHA:
        raise AssertionError("frozen preregistration changed")
    manifest = json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["source"].values():
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise AssertionError(f"source hash changed: {path}")

    candidate = pd.read_parquet(OUT_DIR / "oof/stage1_2023__candidate.parquet")
    weighted = pd.read_parquet(OUT_DIR / "oof/stage1_2023__weighted_recipe.parquet")
    baseline = pd.read_parquet(PROJECT_DIR / "artifacts/oof/dev2023_locked_v3.parquet")
    for frame in (candidate, weighted, baseline):
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    max_formula_diff = 0.0
    formula_bit_exact = True
    for group in GROUPS:
        expected = np.clip(
            0.5 * baseline[group].to_numpy(dtype=float)
            + 0.5 * weighted[group].to_numpy(dtype=float),
            0.0,
            1.02 * CAPACITY[group],
        )
        observed = candidate[group].to_numpy(dtype=float)
        max_formula_diff = max(max_formula_diff, float(np.max(np.abs(expected - observed))))
        formula_bit_exact &= np.array_equal(expected, observed)
    if not formula_bit_exact:
        raise AssertionError("saved candidate differs from frozen formula")

    label_path = Path(r"data/local/open/train/train_labels.csv")
    with label_path.open("rb", buffering=0) as stream:
        payload = stream.read(742551)
        position = stream.tell()
    if position != 742551 or hashlib.sha256(payload).hexdigest() != "ee9707cc81229db760fc035d0bf1d10eb3586d2fef95849bdf463d73fae3aacd":
        raise AssertionError("independent bounded label prefix changed")
    import io

    labels = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    times = pd.to_datetime(labels.pop("kst_dtm"), errors="raise")
    labels.index = pd.DatetimeIndex(times, name="forecast_kst_dtm")
    results = json.loads((OUT_DIR / "stage1_results.json").read_text(encoding="utf-8"))
    masks = segment_masks(candidate.index)
    max_metric_diff = 0.0
    for group in GROUPS:
        actual = labels.loc[candidate.index, group].to_numpy(dtype=float)
        for segment, mask in masks.items():
            base = independent_metric(actual[mask], baseline[group].to_numpy(dtype=float)[mask], CAPACITY[group])
            cand = independent_metric(actual[mask], candidate[group].to_numpy(dtype=float)[mask], CAPACITY[group])
            saved = results["results"][group]["segments"][segment]
            for key in ("one_minus_nmae", "ficr"):
                max_metric_diff = max(max_metric_diff, abs(base[key] - float(saved["baseline"][key])))
                max_metric_diff = max(max_metric_diff, abs(cand[key] - float(saved["candidate"][key])))
                max_metric_diff = max(max_metric_diff, abs((cand[key] - base[key]) - float(saved["delta"][key])))
            max_metric_diff = max(max_metric_diff, abs((cand["total"] - base["total"]) - float(saved["delta"]["total"])))
    if max_metric_diff > 1e-12:
        raise AssertionError(f"independent metric mismatch: {max_metric_diff}")

    weight_audit: dict[str, object] = {}
    for group in GROUPS:
        weights = pd.read_parquet(OUT_DIR / f"weights/stage1_2023__{group}.parquet")
        source_actual = labels.loc[weights.index, group].to_numpy(dtype=float)
        eligible = np.isfinite(source_actual) & (source_actual >= 0.10 * CAPACITY[group])
        eligible_weight = weights["density_weight"].to_numpy(dtype=float)[eligible]
        mean = float(eligible_weight.mean())
        ess = float(eligible_weight.sum() ** 2 / np.square(eligible_weight).sum())
        saved_domain = results["results"][group]["diagnostics"]["domain"]
        if abs(mean - 1.0) > 1e-12 or abs(ess - float(saved_domain["eligible_weight_ess"])) > 1e-9:
            raise AssertionError("weight normalization/ESS audit failed")
        if int(saved_domain["run_fold_overlap"]) != 0 or any(
            int(record["fit_holdout_run_overlap"]) != 0 for record in saved_domain["folds"]
        ):
            raise AssertionError("run-fold overlap audit failed")
        weight_audit[group] = {
            "eligible_mean": mean,
            "eligible_ess": ess,
            "run_fold_overlap": 0,
            "domain_oof_auc": float(saved_domain["domain_oof_auc"]),
        }

    forbidden = list(OUT_DIR.glob("*.csv")) + list((OUT_DIR / "predictions").glob("*.csv")) if (OUT_DIR / "predictions").exists() else list(OUT_DIR.glob("*.csv"))
    if forbidden:
        raise AssertionError(f"rejected experiment created CSV: {forbidden}")
    if (OUT_DIR / "stage2_2024_prescore_lock.json").exists():
        raise AssertionError("Stage2 prescore artifact exists despite Stage1 rejection")
    audit = {
        "schema_version": 1,
        "status": "pass",
        "preregister_sha256": PREREG_SHA,
        "feature_names_sha256": FEATURE_SHA,
        "candidate_formula_bit_exact": formula_bit_exact,
        "candidate_formula_max_abs_kwh": max_formula_diff,
        "independent_metric_max_abs_difference": max_metric_diff,
        "bounded_label_prefix_bytes": 742551,
        "bounded_label_prefix_sha256": hashlib.sha256(payload).hexdigest(),
        "next_2024_label_value_cells_read": 0,
        "weights": weight_audit,
        "stage1_locked_groups": results["locked_groups"],
        "2024_label_values_read": False,
        "2025_weather_or_label_values_read_by_canonical_runner": False,
        "csv_created": False,
        "duplicate_process_ledger_sha256": sha256_file(OUT_DIR / "duplicate_process_ledger.json"),
    }
    destination = OUT_DIR / "independent_audit.json"
    destination.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()

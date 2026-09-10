from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_LABELS = Path(r"data/local/open/train/train_labels.csv")
GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
EXPECTED = {
    "v2": {
        "directory": "direct_interval_selective_scale_transfer_v2",
        "config_sha256": "217ec04c2943bb02fc53a6d8c02ee5ab9c715d9e0d55c0b8b5dae1cbf1391f03",
        "manifest_sha256": "83029de9f30b5f49f8aa4161cdccfbd3891fb80d2b1c29a96914f4d3bb0f0e57",
        "specs": {"kpx_group_1": (0.98, 0.01), "kpx_group_2": (0.98, 0.01)},
    },
    "v3": {
        "directory": "direct_interval_public_adaptive_transfer_v3",
        "config_sha256": "917a197037d3ad1e625805efafa5f5141340546e38a96a18e61da4245383a3fe",
        "manifest_sha256": "a3958124c47c6ada7d38471b5034d3a27ef865941ad9dda8f133b997af752726",
        "specs": {"kpx_group_1": (0.98, 0.0125), "kpx_group_2": (0.95, 0.0025)},
    },
}
ADDENDUM_SHA = "132e4193adf5ce3d4ea4f7f7d4cef83a1f50fe20f7625fd0485a8ba3161e7dee"
V1_MANIFEST_SHA = "0ac3e2b01820d3ca47f449a7efa79c2a7670c35fcf9c61a7ac23c605088a2b0e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(actual: Any, expected: Any, context: str) -> None:
    if isinstance(expected, Mapping):
        if set(actual) != set(expected):
            raise AssertionError(f"{context}: keys differ")
        for key in expected:
            assert_close(actual[key], expected[key], f"{context}/{key}")
    elif isinstance(expected, (int, np.integer)) and not isinstance(expected, bool):
        if int(actual) != int(expected):
            raise AssertionError(f"{context}: {actual} != {expected}")
    elif isinstance(expected, (float, np.floating)):
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=2e-14):
            raise AssertionError(f"{context}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(f"{context}: {actual!r} != {expected!r}")


def year_segments(year: int) -> dict[str, pd.DatetimeIndex]:
    boundaries = {
        "start": pd.Timestamp(f"{year}-01-01 01:00"),
        "q2": pd.Timestamp(f"{year}-04-01 01:00"),
        "h2": pd.Timestamp(f"{year}-07-01 01:00"),
        "q4": pd.Timestamp(f"{year}-10-01 01:00"),
        "end": pd.Timestamp(f"{year + 1}-01-01 00:00"),
    }
    hour = pd.Timedelta(hours=1)
    interval = lambda a, b: pd.date_range(a, b, freq="h", name="forecast_kst_dtm")
    return {
        "full": interval(boundaries["start"], boundaries["end"]),
        "H1": interval(boundaries["start"], boundaries["h2"] - hour),
        "H2": interval(boundaries["h2"], boundaries["end"]),
        "Q1": interval(boundaries["start"], boundaries["q2"] - hour),
        "Q2": interval(boundaries["q2"], boundaries["h2"] - hour),
        "Q3": interval(boundaries["h2"], boundaries["q4"] - hour),
        "Q4": interval(boundaries["q4"], boundaries["end"]),
    }


def group_metrics(actual: np.ndarray, prediction: np.ndarray, group: str) -> dict[str, Any]:
    capacity = CAPACITY[group]
    valid = np.isfinite(actual) & (actual >= 0.10 * capacity)
    y = actual[valid]
    p = prediction[valid]
    if not np.isfinite(p).all():
        raise AssertionError(f"{group}: nonfinite evaluated prediction")
    error = np.abs(p - y) / capacity
    within = error <= 0.06
    between = (~within) & (error <= 0.08)
    over = error > 0.08
    unit = np.select([within, error <= 0.08], [4.0, 3.0], default=0.0)
    energy = float(np.sum(y))
    earned = float(np.sum(y * unit))
    maximum = float(np.sum(y * 4.0))
    nmae = float(np.mean(error))
    share = lambda mask: float(np.sum(y[mask]) / energy)
    ficr = earned / maximum
    result = {
        "group": group,
        "capacity_kwh": capacity,
        "n_evaluated": int(valid.sum()),
        "actual_energy_kwh": energy,
        "nmae": nmae,
        "one_minus_nmae": 1.0 - nmae,
        "ficr": ficr,
        "earned_settlement": earned,
        "max_settlement": maximum,
        "within_6_count": int(within.sum()),
        "between_6_and_8_count": int(between.sum()),
        "over_8_count": int(over.sum()),
        "within_6_energy_share": share(within),
        "between_6_and_8_energy_share": share(between),
        "over_8_energy_share": share(over),
    }
    return result


def group_score(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    result = group_metrics(actual.to_numpy(float), prediction.to_numpy(float), group)
    result["score"] = 0.5 * (result["one_minus_nmae"] + result["ficr"])
    return result


def mixed_score(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    by_group = {
        group: group_metrics(actual[group].to_numpy(float), prediction[group].to_numpy(float), group)
        for group in GROUPS
    }
    one_minus = 1.0 - float(np.mean([1.0 - row["one_minus_nmae"] for row in by_group.values()]))
    ficr = float(np.mean([row["ficr"] for row in by_group.values()]))
    return {"total_score": 0.5 * (one_minus + ficr), "one_minus_nmae": one_minus, "ficr": ficr, "by_group": by_group}


def interpolate(surface: np.ndarray, query: np.ndarray) -> np.ndarray:
    coordinate = 100.0 * np.clip(query, 0.0, 1.02)
    lower = np.floor(coordinate).astype(np.int16)
    upper = np.minimum(lower + 1, 102).astype(np.int16)
    fraction = coordinate - lower
    rows = np.arange(len(query))
    return (1.0 - fraction) * surface[rows, lower] + fraction * surface[rows, upper]


def verify_formula(
    baseline: pd.Series,
    candidate: pd.Series,
    diagnostics: pd.DataFrame,
    utility: np.ndarray,
    group: str,
    factor: float,
    margin: float,
) -> int:
    capacity = CAPACITY[group]
    base = baseline.to_numpy(float)
    base_cf = base / capacity
    scaled_cf = factor * base_cf
    base_utility = interpolate(utility, base_cf)
    scaled_utility = interpolate(utility, scaled_cf)
    advantage = scaled_utility - base_utility
    gate = advantage > margin
    expected = np.clip(np.where(gate, factor * base, base), 0.0, 1.02 * capacity)
    if not np.array_equal(candidate.to_numpy(float), expected):
        raise AssertionError(f"{group}: candidate formula differs")
    expected_columns = {
        "baseline_cf": base_cf,
        "scaled_cf": scaled_cf,
        "base_utility": base_utility,
        "scaled_utility": scaled_utility,
        "utility_advantage": advantage,
        "gate": gate,
        "candidate_cf": expected / capacity,
    }
    if not diagnostics.index.equals(baseline.index):
        raise AssertionError(f"{group}: diagnostic index differs")
    for column, values in expected_columns.items():
        observed = diagnostics[column].to_numpy()
        if not np.array_equal(observed, values, equal_nan=True):
            raise AssertionError(f"{group}: diagnostic {column} differs")
    return int(gate.sum())


def verify_manifest(root: Path, expected_sha: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if sha256(manifest_path) != expected_sha:
        raise AssertionError(f"{root.name}: manifest hash differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = set()
    for record in manifest["outputs"]:
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]) or sha256(path) != record["sha256"]:
            raise AssertionError(f"{root.name}: manifest output differs: {path}")
        listed.add(path.resolve())
    actual = {path.resolve() for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"}
    if listed != actual or len(actual) != int(manifest["output_count_excluding_manifest"]):
        raise AssertionError(f"{root.name}: manifest closure differs")
    for record in manifest["source_config_snapshot"].values():
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]) or sha256(path) != record["sha256"]:
            raise AssertionError(f"{root.name}: source snapshot differs: {path}")
    return manifest


def verify_metrics(
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    stored: Mapping[str, Any],
) -> None:
    segments = year_segments(2024)
    for group in ("kpx_group_1", "kpx_group_2"):
        for name in SEGMENTS:
            index = segments[name]
            record = stored["group_comparisons"][group][name]
            base = group_score(labels.loc[index, group], baseline.loc[index, group], group)
            cand = group_score(labels.loc[index, group], candidate.loc[index, group], group)
            assert_close(record["baseline"], base, f"group/{group}/{name}/baseline")
            assert_close(record["candidate"], cand, f"group/{group}/{name}/candidate")
            assert_close(record["delta"], cand["score"] - base["score"], f"group/{group}/{name}/delta")
    for name in SEGMENTS:
        index = segments[name]
        record = stored["mixed_comparisons"][name]
        base = mixed_score(labels.loc[index], baseline.loc[index])
        cand = mixed_score(labels.loc[index], candidate.loc[index])
        assert_close(record["baseline"], base, f"mixed/{name}/baseline")
        assert_close(record["candidate"], cand, f"mixed/{name}/candidate")
        assert_close(record["delta_total_score"], cand["total_score"] - base["total_score"], f"mixed/{name}/total")
        assert_close(record["delta_one_minus_nmae"], cand["one_minus_nmae"] - base["one_minus_nmae"], f"mixed/{name}/nmae")
        assert_close(record["delta_ficr"], cand["ficr"] - base["ficr"], f"mixed/{name}/ficr")


def main() -> None:
    postgate = ROOT / "artifacts/postgate"
    roots = {variant: postgate / spec["directory"] for variant, spec in EXPECTED.items()}
    before = {variant: sha256(root / "manifest.json") for variant, root in roots.items()}
    if sha256(postgate / "direct_interval_probability_strict_v1/manifest.json") != V1_MANIFEST_SHA:
        raise AssertionError("v1 canonical manifest mutated")
    manifests = {variant: verify_manifest(roots[variant], spec["manifest_sha256"]) for variant, spec in EXPECTED.items()}

    for variant, spec in EXPECTED.items():
        root = roots[variant]
        if sha256(root / "preregister.json") != spec["config_sha256"] or sha256(root / "interaction_addendum.json") != ADDENDUM_SHA:
            raise AssertionError(f"{variant}: config/addendum hash differs")
        if not manifests[variant]["risk"]["selection_unsafe"] or manifests[variant]["risk"]["strict_final_isolation"]:
            raise AssertionError(f"{variant}: risk disclosure differs")
        if not all(manifests[variant]["incidents"].values()):
            raise AssertionError(f"{variant}: incident flags incomplete")

    for group in ("kpx_group_1", "kpx_group_2"):
        for kind in ("models", "surfaces"):
            suffix = "joblib" if kind == "models" else "npz"
            if sha256(roots["v2"] / f"stage2/{kind}/{group}.{suffix}") != sha256(roots["v3"] / f"stage2/{kind}/{group}.{suffix}"):
                raise AssertionError(f"shared {kind}/{group} differs")

    authoritative = {
        "primary_v3": ROOT / "artifacts/oof/gate2024_locked_v3_cf_fix.parquet",
        "interaction_v4": ROOT / "artifacts/oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet",
    }
    labels = pd.read_csv(RAW_LABELS, encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    labels_2024 = labels.loc[year_segments(2024)["full"], list(GROUPS)]
    report: dict[str, Any] = {"schema_version": 1, "status": "PASS", "variants": {}}
    for variant, spec in EXPECTED.items():
        root = roots[variant]
        stage2 = json.loads((root / "stage2_results.json").read_text(encoding="utf-8"))
        variant_report: dict[str, Any] = {"stage2": {}, "stage1_formula_checks": {}}
        utilities = {group: np.load(root / f"stage2/surfaces/{group}.npz")["utility"] for group in ("kpx_group_1", "kpx_group_2")}
        for baseline_name in ("primary_v3", "interaction_v4"):
            baseline = pd.read_parquet(root / f"stage2/{baseline_name}/baseline_2024.parquet")
            candidate = pd.read_parquet(root / f"stage2/{baseline_name}/candidate_2024.parquet")
            source = pd.read_parquet(authoritative[baseline_name]).astype(float)
            source.index = pd.DatetimeIndex(source.index, name="forecast_kst_dtm")
            if not baseline.equals(source):
                raise AssertionError(f"{variant}/{baseline_name}: authoritative baseline values differ")
            gates: dict[str, int] = {}
            for group, (factor, margin) in spec["specs"].items():
                diagnostics = pd.read_parquet(root / f"stage2/{baseline_name}/{group}_diagnostics.parquet")
                gates[group] = verify_formula(baseline[group], candidate[group], diagnostics, utilities[group], group, factor, margin)
            if not np.array_equal(candidate["kpx_group_3"].to_numpy(), baseline["kpx_group_3"].to_numpy()):
                raise AssertionError(f"{variant}/{baseline_name}: G3 is not identity")
            verify_metrics(labels_2024, baseline, candidate, stage2["baseline_results"][baseline_name])
            audit = stage2["baseline_results"][baseline_name]["promotion_audit"]
            group_deltas = audit["group_deltas"]
            mixed_deltas = audit["mixed_total_deltas"]
            expected_pass = all(value > 0 for value in group_deltas.values()) and all(value > 0 for value in mixed_deltas.values()) and audit["full_delta_one_minus_nmae"] >= 0 and audit["full_delta_ficr"] >= 0
            if expected_pass != bool(stage2["baseline_results"][baseline_name]["passed"]):
                raise AssertionError(f"{variant}/{baseline_name}: promotion gate differs")
            variant_report["stage2"][baseline_name] = {
                "candidate_formula_exact": True,
                "gate_counts": gates,
                "comparison_records_recomputed": 21,
                "minimum_group_delta": min(group_deltas.values()),
                "minimum_mixed_delta": min(mixed_deltas.values()),
                "full_delta_one_minus_nmae": audit["full_delta_one_minus_nmae"],
                "full_delta_ficr": audit["full_delta_ficr"],
                "passed": expected_pass,
            }
        if bool(stage2["dual_baseline_promoted"]) != all(row["passed"] for row in variant_report["stage2"].values()):
            raise AssertionError(f"{variant}: dual gate differs")
        final = json.loads((root / "final_results.json").read_text(encoding="utf-8"))
        promotion = json.loads((root / "stage2_promotion_lock.json").read_text(encoding="utf-8"))
        if final["csv_created"] or final["executed"] or promotion["csv_allowed"] or list(root.rglob("*.csv")):
            raise AssertionError(f"{variant}: rejected branch created/allowed CSV")
        variant_report.update(
            {
                "stage2_comparison_records_recomputed": 42,
                "primary_passed": bool(stage2["primary_passed"]),
                "interaction_passed": bool(stage2["interaction_passed"]),
                "dual_baseline_promoted": bool(stage2["dual_baseline_promoted"]),
                "no_csv": True,
                "incident_and_risk_disclosure": True,
            }
        )
        report["variants"][variant] = variant_report

    # Stage-1 formula audit uses the immutable v1 surfaces, including the v2-only G3 test.
    v1 = postgate / "direct_interval_probability_strict_v1"
    for variant, spec in EXPECTED.items():
        root = roots[variant]
        baseline = pd.read_parquet(root / "stage1/baseline_2023.parquet")
        candidate = pd.read_parquet(root / "stage1/candidate_2023.parquet")
        for group in GROUPS:
            index = baseline.index if group != "kpx_group_3" else year_segments(2023)["H2"]
            diagnostics = pd.read_parquet(root / f"stage1/{group}_diagnostics.parquet")
            with np.load(v1 / f"oof/stage1_{group}_surfaces.npz") as arrays:
                utility = arrays["utility"]
            if variant == "v2" and group == "kpx_group_3":
                factor, margin = (0.98, 0.01)
                gates = verify_formula(baseline.loc[index, group], candidate.loc[index, group], diagnostics, utility, group, factor, margin)
            elif group in spec["specs"]:
                factor, margin = spec["specs"][group]
                gates = verify_formula(baseline.loc[index, group], candidate.loc[index, group], diagnostics, utility, group, factor, margin)
            else:
                gates = 0
                if not np.array_equal(candidate.loc[index, group].to_numpy(), baseline.loc[index, group].to_numpy()):
                    raise AssertionError(f"{variant}/stage1/{group}: identity differs")
            report["variants"][variant]["stage1_formula_checks"][group] = {"gate_count": gates, "exact": True}

    after = {variant: sha256(root / "manifest.json") for variant, root in roots.items()}
    if before != after:
        raise AssertionError("canonical manifests changed during read-only audit")
    report["canonical_nonmutation"] = True
    report["shared_stage2_models_and_surfaces_bit_exact"] = True
    report["v1_manifest_untouched"] = True
    report["manifest_hashes"] = after
    destination = ROOT / "artifacts/audits/direct_interval_selective_transfer_v2_v3_independent.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(f"PASS {destination} sha256={sha256(destination)}")


if __name__ == "__main__":
    main()

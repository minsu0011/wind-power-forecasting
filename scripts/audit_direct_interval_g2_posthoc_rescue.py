"""Independent read-only audit for the G2-only posthoc rescue canonical."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.manifest import sha256_file, utc_now  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


CANONICAL = PROJECT_DIR / "artifacts/postgate/direct_interval_g2_posthoc_rescue_v4"
CONFIG = PROJECT_DIR / "configs/direct_interval_g2_posthoc_rescue_preregister_v4.json"
EXPECTED_CONFIG_SHA = "5cf54b2aad90ca8d2edc70dbc891eeafe8dc0a70804925354d33ebca64e1b45f"
GROUP = "kpx_group_2"
IDENTITY = ("kpx_group_1", "kpx_group_3")
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _verify_description(spec: dict[str, Any]) -> Path:
    path = _path(str(spec["path"]))
    registered_size = spec.get("size_bytes", spec.get("bytes"))
    if registered_size is not None and path.stat().st_size != int(registered_size):
        raise AssertionError(f"size mismatch: {path}")
    if sha256_file(path) != spec["sha256"]:
        raise AssertionError(f"hash mismatch: {path}")
    return path


def _bits(series: pd.Series) -> np.ndarray:
    return np.ascontiguousarray(series.to_numpy(dtype=np.float64)).view(np.uint64)


def _interp(utility: np.ndarray, query: np.ndarray) -> np.ndarray:
    q = 100.0 * np.clip(np.asarray(query, dtype=np.float64), 0.0, 1.02)
    lower = np.floor(q).astype(np.int64)
    upper = np.minimum(lower + 1, 102)
    fraction = q - lower
    row = np.arange(len(q), dtype=np.int64)
    return (1.0 - fraction) * utility[row, lower] + fraction * utility[row, upper]


def _group_values(actual: pd.Series, forecast: pd.Series, group: str) -> tuple[float, float, float]:
    capacity = CAPACITY_KWH[group]
    y = actual.to_numpy(dtype=np.float64)
    p = forecast.to_numpy(dtype=np.float64)
    eligible = np.isfinite(y) & (y >= 0.10 * capacity)
    y, p = y[eligible], p[eligible]
    error = np.abs(p - y) / capacity
    n = 1.0 - float(error.mean())
    unit_price = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    f = float(np.sum(y * unit_price) / np.sum(y * 4.0))
    return 0.5 * (n + f), n, f


def _mixed_values(actual: pd.DataFrame, forecast: pd.DataFrame) -> tuple[float, float, float]:
    records = [_group_values(actual[group], forecast[group], group) for group in TARGET_COLS]
    n = float(np.mean([record[1] for record in records]))
    f = float(np.mean([record[2] for record in records]))
    return 0.5 * (n + f), n, f


def _segments(year: int) -> dict[str, pd.DatetimeIndex]:
    def interval(start: str, end: str) -> pd.DatetimeIndex:
        return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")
    full = interval(f"{year}-01-01 01:00", f"{year + 1}-01-01 00:00")
    return {
        "full": full,
        "H1": interval(f"{year}-01-01 01:00", f"{year}-07-01 00:00"),
        "H2": interval(f"{year}-07-01 01:00", f"{year + 1}-01-01 00:00"),
        "Q1": interval(f"{year}-01-01 01:00", f"{year}-04-01 00:00"),
        "Q2": interval(f"{year}-04-01 01:00", f"{year}-07-01 00:00"),
        "Q3": interval(f"{year}-07-01 01:00", f"{year}-10-01 00:00"),
        "Q4": interval(f"{year}-10-01 01:00", f"{year + 1}-01-01 00:00"),
    }


def main() -> None:
    before = {
        path.relative_to(CANONICAL).as_posix(): sha256_file(path)
        for path in CANONICAL.rglob("*") if path.is_file()
    }
    if sha256_file(CONFIG) != EXPECTED_CONFIG_SHA:
        raise AssertionError("config hash differs")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest_path = CANONICAL / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != EXPECTED_CONFIG_SHA:
        raise AssertionError("manifest config binding differs")
    if not manifest["risk"]["posthoc_2024_rescue"] or not manifest["risk"]["selection_unsafe"]:
        raise AssertionError("risk disclosure missing")
    if manifest["risk"]["strict_final_isolation"]:
        raise AssertionError("invalid strict isolation claim")

    output_paths = set()
    for item in manifest["outputs"]:
        path = _verify_description(item)
        if path.parent != CANONICAL and CANONICAL not in path.parents:
            raise AssertionError("manifest output escapes canonical")
        output_paths.add(path.relative_to(CANONICAL).as_posix())
    actual_paths = {
        path.relative_to(CANONICAL).as_posix()
        for path in CANONICAL.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if output_paths != actual_paths:
        raise AssertionError("manifest output closure differs")
    for snapshot in manifest["source_config_snapshot"].values():
        _verify_description(snapshot)
    for spec in config["lineage"].values():
        _verify_description(spec)

    labels_spec = config["final_2025_contract"]["labels"]
    labels_path = _verify_description(labels_spec)
    labels = pd.read_csv(labels_path)
    labels.index = pd.DatetimeIndex(pd.to_datetime(labels.pop("kst_dtm")), name="forecast_kst_dtm")
    labels = labels.loc[:, list(TARGET_COLS)].astype(np.float64)
    expected = config["frozen_2024_posthoc_rescue_evidence"]
    year_segments = _segments(2024)
    stage2_checks: dict[str, Any] = {}
    for name in ("primary_v3", "interaction_v4"):
        root = CANONICAL / f"rescue_2024/{name}"
        baseline = pd.read_parquet(root / "baseline.parquet").astype(np.float64)
        candidate = pd.read_parquet(root / "candidate.parquet").astype(np.float64)
        detail = pd.read_parquet(root / "g2_diagnostics.parquet")
        for group in IDENTITY:
            if not np.array_equal(_bits(candidate[group]), _bits(baseline[group])):
                raise AssertionError(f"2024 identity differs: {name}/{group}")
        surface_path = _verify_description(config["locked_2024_artifacts"]["shared_g2_surface"])
        with np.load(surface_path) as payload:
            utility = np.asarray(payload["utility"], dtype=np.float64)
        base_cf = baseline[GROUP].to_numpy() / CAPACITY_KWH[GROUP]
        advantage = _interp(utility, 0.98 * base_cf) - _interp(utility, base_cf)
        gate = advantage > 0.01
        expected_g2 = np.clip(np.where(gate, 0.98 * baseline[GROUP].to_numpy(), baseline[GROUP].to_numpy()), 0.0, 1.02 * CAPACITY_KWH[GROUP])
        if not np.array_equal(np.ascontiguousarray(expected_g2).view(np.uint64), _bits(candidate[GROUP])):
            raise AssertionError(f"2024 G2 formula differs: {name}")
        if not np.array_equal(gate, detail["gate"].to_numpy(dtype=bool)):
            raise AssertionError(f"2024 G2 gate differs: {name}")
        minima: dict[str, float] = {}
        for segment_name in SEGMENTS:
            index = year_segments[segment_name]
            gb = _group_values(labels.loc[index, GROUP], baseline.loc[index, GROUP], GROUP)
            gc = _group_values(labels.loc[index, GROUP], candidate.loc[index, GROUP], GROUP)
            mb = _mixed_values(labels.loc[index], baseline.loc[index])
            mc = _mixed_values(labels.loc[index], candidate.loc[index])
            gd = gc[0] - gb[0]
            md = mc[0] - mb[0]
            if not np.isclose(gd, expected["g2_group_delta_total_score"][name][segment_name], atol=2e-15, rtol=0):
                raise AssertionError(f"2024 G2 metric differs: {name}/{segment_name}")
            if not np.isclose(md, expected["mixed_delta_total_score"][name][segment_name], atol=2e-15, rtol=0):
                raise AssertionError(f"2024 mixed metric differs: {name}/{segment_name}")
            if gd <= 0 or md <= 0:
                raise AssertionError(f"2024 positive gate differs: {name}/{segment_name}")
            minima[segment_name] = md
            if segment_name == "full":
                components = expected["mixed_full_component_deltas"][name]
                if not np.isclose(mc[1] - mb[1], components["one_minus_nmae"], atol=2e-15, rtol=0):
                    raise AssertionError("2024 NMAE component differs")
                if not np.isclose(mc[2] - mb[2], components["ficr"], atol=2e-15, rtol=0):
                    raise AssertionError("2024 FICR component differs")
        stage2_checks[name] = {"selected_rows": int(gate.sum()), "min_mixed_delta": min(minima.values()), "segments_passed": 7}

    final = config["final_2025_contract"]
    baseline_path = _verify_description(final["baseline"])
    baseline = pd.read_parquet(baseline_path).astype(np.float64)
    prediction = pd.read_parquet(CANONICAL / "final/predictions.parquet").astype(np.float64)
    detail = pd.read_parquet(CANONICAL / "final/g2_diagnostics.parquet")
    with np.load(CANONICAL / "final/surface/kpx_group_2.npz") as payload:
        surface = {key: np.asarray(payload[key]) for key in payload.files}
    if not np.all(surface["p6"] <= surface["p8"]):
        raise AssertionError("p6<=p8 repair invariant fails")
    utility = surface["utility"]
    base_cf = baseline[GROUP].to_numpy() / CAPACITY_KWH[GROUP]
    advantage = _interp(utility, 0.98 * base_cf) - _interp(utility, base_cf)
    gate = advantage > 0.01
    expected_g2 = np.clip(np.where(gate, 0.98 * baseline[GROUP].to_numpy(), baseline[GROUP].to_numpy()), 0.0, 1.02 * CAPACITY_KWH[GROUP])
    if not np.array_equal(np.ascontiguousarray(expected_g2).view(np.uint64), _bits(prediction[GROUP])):
        raise AssertionError("final G2 formula differs")
    if not np.array_equal(gate, detail["gate"].to_numpy(dtype=bool)) or int(gate.sum()) != 143:
        raise AssertionError("final gate/count differs")
    for group in IDENTITY:
        if not np.array_equal(_bits(prediction[group]), _bits(baseline[group])):
            raise AssertionError(f"final parquet identity differs: {group}")

    sample_path = _verify_description(final["sample"])
    sample = pd.read_csv(sample_path, encoding="utf-8-sig", dtype="string")
    csv_path = CANONICAL / str(final["candidate_csv"])
    raw = csv_path.read_bytes()
    if not raw.startswith(b"\xef\xbb\xbf"):
        raise AssertionError("CSV BOM missing")
    csv = pd.read_csv(csv_path, encoding="utf-8-sig", dtype="string")
    if tuple(csv.columns) != tuple(sample.columns) or len(csv) != 8760:
        raise AssertionError("CSV schema/rows differ")
    if not csv[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]]):
        raise AssertionError("CSV sample ID/time differs")
    for group in TARGET_COLS:
        expected_text = prediction[group].reset_index(drop=True).map(lambda value: f"{float(value):.6f}").astype("string")
        if not csv[group].equals(expected_text):
            raise AssertionError(f"CSV six-decimal values differ: {group}")
    numeric = csv.loc[:, list(TARGET_COLS)].astype(np.float64).to_numpy()
    if not np.isfinite(numeric).all():
        raise AssertionError("CSV non-finite values")
    for position, group in enumerate(TARGET_COLS):
        if numeric[:, position].min() < 0 or numeric[:, position].max() > 1.02 * CAPACITY_KWH[group] + 5e-7:
            raise AssertionError(f"CSV bounds differ: {group}")

    lock = json.loads((CANONICAL / "final_prescore_lock.json").read_text(encoding="utf-8"))
    if any(lock[key] != 0 for key in (
        "test_weather_value_cells_parsed_before_lock",
        "final_baseline_prediction_value_cells_parsed_before_lock",
        "sample_value_bytes_read_before_lock",
    )):
        raise AssertionError("prescore access-order record differs")
    after = {
        path.relative_to(CANONICAL).as_posix(): sha256_file(path)
        for path in CANONICAL.rglob("*") if path.is_file()
    }
    if before != after:
        raise AssertionError("audit mutated canonical")
    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "audit_type": "independent_read_only_direct_interval_g2_posthoc_rescue_v4",
        "status": "PASS",
        "canonical": str(CANONICAL.resolve()),
        "config_sha256": EXPECTED_CONFIG_SHA,
        "manifest_sha256": sha256_file(manifest_path),
        "csv_sha256": sha256_file(csv_path),
        "checks": {
            "manifest_output_and_source_closure": True,
            "lineage_hashes": True,
            "risk_and_incident_disclosure": True,
            "2024_two_baseline_formula_metrics_and_gates": stage2_checks,
            "final_linear_formula_strict_margin": True,
            "final_g2_selected_rows": int(gate.sum()),
            "p6_le_p8": True,
            "g1_g3_float64_parquet_identity": True,
            "csv_sample_schema_rows_time_bom_six_decimals_finite_bounds": True,
            "prescore_access_order_record": True,
            "canonical_nonmutation": True,
        },
        "limitations": {
            "posthoc_2024_rescue": True,
            "selection_unsafe": True,
            "strict_final_isolation": False,
            "leaderboard_score_claim": False,
        },
        "canonical_hashes_before_after_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
    }
    out = PROJECT_DIR / "artifacts/audits/direct_interval_g2_posthoc_rescue_v4_independent.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS {out} sha256={sha256_file(out)}")


if __name__ == "__main__":
    main()

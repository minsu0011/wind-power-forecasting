"""Independent formula/metric audit of neural interval-utility transfer v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_neural_interval_utility_transfer as producer  # noqa: E402

CANONICAL = ROOT / "artifacts/postgate/neural_interval_utility_transfer_v2"
CONFIG = ROOT / "configs/neural_interval_utility_transfer_preregister_v2.json"
CONFIG_SHA = "8cc4f547662e4550eb50860f6b2d987445e57f161f5424e1ac1426c4984fbcc1"
LOCK_SHA = "7cef92525ed5df3a97fde7ad5998b79f10bdfc4efae3e88949e98606456987ad"
MANIFEST_SHA = "8387b98e7f426362f1db8fe23b83561e49c1a2dd8187f04ed69d2ce2a697b119"
GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
ACTIVE = GROUPS[:2]
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/audits/neural_interval_utility_transfer_v2_local.json"))
    return parser.parse_args(argv)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify(spec: Mapping[str, Any]) -> Path:
    path = Path(str(spec["path"]))
    path = path if path.is_absolute() else ROOT / path
    assert path.is_file()
    size = spec.get("bytes", spec.get("size_bytes"))
    if size is not None:
        assert path.stat().st_size == int(size)
    assert sha(path) == str(spec["sha256"])
    return path


def bits(series: pd.Series) -> np.ndarray:
    return np.ascontiguousarray(series.to_numpy(dtype=np.float64)).view(np.uint64)


def interpolation(surface: np.ndarray, query: np.ndarray) -> np.ndarray:
    position = np.clip(query, 0.0, 1.02) * 100.0
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, 102)
    fraction = position - lower
    row = np.arange(len(query))
    return (1.0 - fraction) * surface[row, lower] + fraction * surface[row, upper]


def segments() -> dict[str, pd.DatetimeIndex]:
    def rows(start: str, end: str) -> pd.DatetimeIndex:
        return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")
    return {
        "full": rows("2024-01-01 01:00", "2025-01-01 00:00"),
        "H1": rows("2024-01-01 01:00", "2024-07-01 00:00"),
        "H2": rows("2024-07-01 01:00", "2025-01-01 00:00"),
        "Q1": rows("2024-01-01 01:00", "2024-04-01 00:00"),
        "Q2": rows("2024-04-01 01:00", "2024-07-01 00:00"),
        "Q3": rows("2024-07-01 01:00", "2024-10-01 00:00"),
        "Q4": rows("2024-10-01 01:00", "2025-01-01 00:00"),
    }


def metric(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, float]:
    capacity = CAPACITY[group]
    y = actual.to_numpy(dtype=np.float64)
    p = prediction.to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & (y >= 0.10 * capacity)
    y, p = y[valid], p[valid]
    error = np.abs(p - y) / capacity
    n = 1.0 - float(error.mean())
    points = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    f = float(np.sum(y * points) / np.sum(y * 4.0))
    return {"total_score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f}


def mixed(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, float]:
    records = [metric(actual[group], prediction[group], group) for group in GROUPS]
    n = float(np.mean([record["one_minus_nmae"] for record in records]))
    f = float(np.mean([record["ficr"] for record in records]))
    return {"total_score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f}


def close(left: float, right: float) -> None:
    assert np.isclose(left, right, rtol=0.0, atol=1e-15), (left, right)


def write(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = path if path.is_absolute() else ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    canonical_before = {
        path.relative_to(CANONICAL).as_posix(): (path.stat().st_size, sha(path))
        for path in CANONICAL.rglob("*")
        if path.is_file()
    }
    assert sha(CONFIG) == CONFIG_SHA
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert sha(CANONICAL / "candidate_before_2024_label_lock.json") == LOCK_SHA
    assert sha(CANONICAL / "manifest.json") == MANIFEST_SHA
    manifest = json.loads((CANONICAL / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["risk"]["selection_unsafe"] is True
    assert manifest["risk"]["private_champion"] is False
    assert manifest["stage2"]["dual_baseline_promoted"] is False
    manifest_paths = {verify(spec).resolve() for spec in manifest["outputs"]}
    actual_output_paths = {
        path.resolve()
        for path in CANONICAL.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert manifest_paths == actual_output_paths
    assert len(manifest_paths) == manifest["output_count_excluding_manifest"] == 17
    for spec in config["lineage"].values():
        verify(spec)
    for spec in config["input_identities"].values():
        verify(spec)
    for spec in manifest["outputs"]:
        verify(spec)
    prescore = json.loads((CANONICAL / "stage2_prescore_record.json").read_text(encoding="utf-8"))
    for spec in prescore["outputs"]:
        verify(spec)
    for spec in prescore["upstream_before"].values():
        verify(spec)
    closure = producer.resolve_ast_closure(
        ROOT / "scripts/run_neural_interval_utility_transfer.py"
    )
    closure_relative = [path.relative_to(ROOT).as_posix() for path in closure]
    assert closure_relative == prescore["closure"]["resolved_relative_paths"]
    assert len(closure) == prescore["closure"]["resolved_file_count"] == 17
    assert {path.resolve() for path in closure} == {
        verify(spec).resolve() for spec in prescore["closure"]["resolved_files"]
    }
    verify(prescore["closure"]["test"])
    verify(prescore["closure"]["config"])
    verify(prescore["closure"]["sidecar"])
    candidate_lock = json.loads(
        (CANONICAL / "candidate_before_2024_label_lock.json").read_text(encoding="utf-8")
    )
    assert candidate_lock["created_before_2024_application_labels_were_reopened_for_this_run"]
    assert candidate_lock["candidate_model_gate_delta_both_baselines_frozen"]
    assert candidate_lock["no_metric_values_computed"]
    assert {verify(spec).resolve() for spec in candidate_lock["candidate_outputs"]} == {
        verify(spec).resolve() for spec in prescore["outputs"]
    }

    index = segments()["full"]
    primary = pd.read_parquet(CANONICAL / "stage2/primary_baseline_2024.parquet").astype(np.float64)
    interaction = pd.read_parquet(CANONICAL / "stage2/recent_v4_baseline_2024.parquet").astype(np.float64)
    saved_primary = pd.read_parquet(CANONICAL / "stage2/primary_candidate_2024.parquet").astype(np.float64)
    saved_interaction = pd.read_parquet(CANONICAL / "stage2/recent_v4_candidate_2024.parquet").astype(np.float64)
    saved_delta = pd.read_parquet(CANONICAL / "stage2/primary_delta_2024.parquet").astype(np.float64)
    raw = pd.read_parquet(CANONICAL / "stage2/raw_neural_cf_2024.parquet").astype(np.float64)
    for frame in (primary, interaction, saved_primary, saved_interaction, saved_delta, raw):
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        assert frame.index.equals(index)
    expected_primary = primary.copy()
    expected_interaction = interaction.copy()
    expected_delta = pd.DataFrame(0.0, index=index, columns=GROUPS)
    gates: dict[str, int] = {}
    for group, key in zip(ACTIVE, ("stage2_G1_utility_surface", "stage2_G2_utility_surface"), strict=True):
        with np.load(verify(config["input_identities"][key])) as payload:
            utility = np.asarray(payload["utility"], dtype=np.float64)
        base_cf = primary[group].to_numpy() / CAPACITY[group]
        action_cf = base_cf + 0.05 * (np.clip(raw[group].to_numpy(), 0.0, 1.02) - base_cf)
        advantage = interpolation(utility, action_cf) - interpolation(utility, base_cf)
        gate = advantage > 0.01
        chosen = np.clip(action_cf, 0.0, 1.02) * CAPACITY[group]
        delta = chosen - primary[group].to_numpy()
        expected_primary.loc[gate, group] = chosen[gate]
        expected_interaction.loc[gate, group] = np.clip(interaction[group].to_numpy()[gate] + delta[gate], 0.0, 1.02 * CAPACITY[group])
        expected_delta.loc[gate, group] = delta[gate]
        gates[group] = int(gate.sum())
        diagnostics = pd.read_parquet(CANONICAL / f"stage2/{group}_diagnostics.parquet")
        assert np.array_equal(diagnostics["gate"].to_numpy(dtype=bool), gate)
        assert np.array_equal(bits(diagnostics["primary_delta_kwh"]), bits(expected_delta[group]))
    for expected, saved in ((expected_primary, saved_primary), (expected_interaction, saved_interaction), (expected_delta, saved_delta)):
        for group in GROUPS:
            assert np.array_equal(bits(expected[group]), bits(saved[group]))
    assert np.array_equal(bits(primary[GROUPS[2]]), bits(saved_primary[GROUPS[2]]))
    assert np.array_equal(bits(interaction[GROUPS[2]]), bits(saved_interaction[GROUPS[2]]))

    label_path = verify(config["input_identities"]["train_labels"])
    labels = pd.read_csv(label_path)
    labels.index = pd.DatetimeIndex(pd.to_datetime(labels.pop("kst_dtm")), name="forecast_kst_dtm")
    labels = labels.loc[index, list(GROUPS)].astype(np.float64)
    stored = json.loads((CANONICAL / "stage2_results.json").read_text(encoding="utf-8"))
    summary: dict[str, Any] = {}
    for baseline_name, baseline, candidate in (
        ("primary_v3", primary, expected_primary),
        ("recent_v4_interaction", interaction, expected_interaction),
    ):
        all_positive = True
        mins: dict[str, float] = {group: float("inf") for group in ACTIVE}
        mins["mixed"] = float("inf")
        for name, rows in segments().items():
            record = stored[baseline_name]["comparisons"][name]
            for group in ACTIVE:
                before = metric(labels.loc[rows, group], baseline.loc[rows, group], group)
                after = metric(labels.loc[rows, group], candidate.loc[rows, group], group)
                delta = after["total_score"] - before["total_score"]
                close(delta, record["groups"][group]["delta_total_score"])
                mins[group] = min(mins[group], delta)
                all_positive = all_positive and delta > 0.0
            before_mixed = mixed(labels.loc[rows], baseline.loc[rows])
            after_mixed = mixed(labels.loc[rows], candidate.loc[rows])
            delta_mixed = after_mixed["total_score"] - before_mixed["total_score"]
            close(delta_mixed, record["mixed_delta_total_score"])
            mins["mixed"] = min(mins["mixed"], delta_mixed)
            all_positive = all_positive and delta_mixed > 0.0
        full = stored[baseline_name]["comparisons"]["full"]
        passed = bool(all_positive and full["mixed_delta_one_minus_nmae"] >= 0.0 and full["mixed_delta_ficr"] >= 0.0)
        assert passed is stored[baseline_name]["passed"]
        summary[baseline_name] = {"passed": passed, "minimum_deltas": mins}
    assert summary["primary_v3"]["passed"] is False
    assert summary["recent_v4_interaction"]["passed"] is False
    final = json.loads((CANONICAL / "final_results.json").read_text(encoding="utf-8"))
    assert final["executed"] is False and final["CSV_created"] is False
    assert not (CANONICAL / "final").exists()
    assert not list(CANONICAL.glob("*.csv"))
    assert not any("test" in path.relative_to(CANONICAL).parts for path in CANONICAL.rglob("*"))
    canonical_after = {
        path.relative_to(CANONICAL).as_posix(): (path.stat().st_size, sha(path))
        for path in CANONICAL.rglob("*")
        if path.is_file()
    }
    assert canonical_after == canonical_before

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "read_only": True,
        "config_sha256": CONFIG_SHA,
        "candidate_lock_sha256": LOCK_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "formula_gate_delta_both_baselines_exact": True,
        "gate_rows": gates,
        "G3_each_baseline_bit_identity": True,
        "all_42_group_and_mixed_segment_records_recomputed": True,
        "stage2": summary,
        "dual_baseline_promoted": False,
        "final_test_sample_CSV_absent": True,
        "source_input_output_hashes_exact": True,
        "manifest_output_set_exact": True,
        "recursive_AST_closure_exact": True,
        "candidate_lock_before_label_and_metric_verified": True,
        "canonical_nonmutation": True,
    }
    output = write(args.out, payload)
    print(f"status=PASS audit_sha256={sha(output)}")


if __name__ == "__main__":
    main()

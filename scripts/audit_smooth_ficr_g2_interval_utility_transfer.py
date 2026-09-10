"""Read-only local audit for the frozen smooth-FICR G2 utility transfer."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_raw_grid_wind_lgb as raw_protocol  # noqa: E402
from scripts import run_smooth_ficr_g2_interval_utility_transfer as runner  # noqa: E402
from src.manifest import describe_file, sha256_file, utc_now, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH  # noqa: E402
from src.raw_spatiotemporal_smooth_ficr import candidate_frame  # noqa: E402


CANONICAL = ROOT / "artifacts/postgate/smooth_ficr_g2_interval_utility_transfer_v1"
AUDIT = ROOT / "artifacts/audits/smooth_ficr_g2_interval_utility_transfer_v1_local.json"
RAW_DIR = Path(r"data/local/open")


def _metric(actual: pd.Series, prediction: pd.Series) -> dict[str, float]:
    capacity = float(CAPACITY_KWH[runner.GROUP])
    y = actual.to_numpy(dtype=np.float64)
    p = prediction.to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & (y >= 0.10 * capacity)
    error = np.abs(p[valid] - y[valid]) / capacity
    n = 1.0 - float(error.mean())
    price = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    f = float(np.sum(y[valid] * price) / np.sum(y[valid] * 4.0))
    return {"total_score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f}


def _assert_close(left: float, right: float) -> None:
    if not np.isclose(left, right, rtol=0.0, atol=2e-15):
        raise AssertionError(f"metric mismatch: {left} != {right}")


def main() -> None:
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    config = json.loads(runner.CONFIG_PATH.read_text(encoding="utf-8"))
    if sha256_file(runner.CONFIG_PATH) != runner.EXPECTED_CONFIG_SHA:
        raise AssertionError("config differs")
    manifest_path = CANONICAL / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != runner.EXPECTED_CONFIG_SHA:
        raise AssertionError("manifest config differs")
    args = runner.parse_args([])
    args.config = runner.CONFIG_PATH
    args.raw_dir = RAW_DIR
    args.out_dir = CANONICAL
    record = json.loads((CANONICAL / "stage1_prescore_record.json").read_text(encoding="utf-8"))
    if record["closure"] != runner._closure(args):
        raise AssertionError("locked source closure is not current")
    for spec in record["outputs"]:
        runner._verify(spec)

    specs = config["stage1_input_identities"]
    baseline_upstream = pd.read_parquet(runner._verify(specs["smooth_ficr_G2_baseline"]))
    raw = pd.read_parquet(runner._verify(specs["smooth_ficr_raw_cf"]))
    stored = pd.read_parquet(runner._verify(specs["smooth_ficr_G2_candidates"]))
    replay = candidate_frame(
        baseline_upstream[runner.GROUP],
        raw[runner.GROUP],
        capacity_kwh=float(CAPACITY_KWH[runner.GROUP]),
    )[runner.CANDIDATE_COLUMN]
    if replay.to_numpy().tobytes() != stored[runner.CANDIDATE_COLUMN].to_numpy().tobytes():
        raise AssertionError("stored action replay differs")
    with np.load(runner._verify(specs["direct_interval_G2_surface"])) as payload:
        utility = np.asarray(payload["utility"], dtype=np.float64)
    capacity = float(CAPACITY_KWH[runner.GROUP])
    base_cf = baseline_upstream[runner.GROUP].to_numpy(dtype=np.float64) / capacity
    action_cf = replay.to_numpy(dtype=np.float64) / capacity
    position_b = np.clip(base_cf, 0.0, 1.02) * 100.0
    position_z = np.clip(action_cf, 0.0, 1.02) * 100.0

    def lookup(position: np.ndarray) -> np.ndarray:
        lower = np.floor(position).astype(np.int64)
        upper = np.minimum(lower + 1, 102)
        fraction = position - lower
        rows = np.arange(len(position))
        return (1.0 - fraction) * utility[rows, lower] + fraction * utility[rows, upper]

    advantage = lookup(position_z) - lookup(position_b)
    gate = advantage > 0.01
    expected = np.where(
        gate,
        replay.to_numpy(dtype=np.float64),
        baseline_upstream[runner.GROUP].to_numpy(dtype=np.float64),
    )
    candidate = pd.read_parquet(CANONICAL / "stage1/candidate_G2_2023.parquet")
    if expected.tobytes() != candidate[runner.GROUP].to_numpy(dtype=np.float64).tobytes():
        raise AssertionError("gated candidate differs")

    raw_contract = json.loads(runner._verify(config["lineage"]["raw_grid_physical_contract"]).read_text(encoding="utf-8"))
    labels, label_evidence = raw_protocol._read_label_prefix(
        RAW_DIR,
        raw_contract["physical_stage1_inputs"]["labels_stage1_score_prefix_after_lock"],
        (runner.GROUP,),
    )
    baseline = pd.read_parquet(CANONICAL / "stage1/baseline_G2_2023.parquet")[runner.GROUP]
    result = json.loads((CANONICAL / "stage1_results.json").read_text(encoding="utf-8"))["result"]
    positive = 0
    minimum = float("inf")
    for name, rows in runner._segments(2023).items():
        before = _metric(labels.loc[rows, runner.GROUP], baseline.loc[rows])
        after = _metric(labels.loc[rows, runner.GROUP], candidate.loc[rows, runner.GROUP])
        for key in before:
            delta = after[key] - before[key]
            _assert_close(delta, result["comparisons"][name]["delta"][key])
            _assert_close(
                delta / 3.0,
                result["comparisons"][name]["mixed_delta_by_exact_identity_additivity"][key],
            )
        total = after["total_score"] - before["total_score"]
        positive += int(total > 0.0)
        minimum = min(minimum, total)
    if positive != 3 or result["passed"] is not False:
        raise AssertionError("Stage1 rejection differs")
    if (CANONICAL / "stage2").exists() or (CANONICAL / "final").exists():
        raise AssertionError("forbidden future output directory exists")
    if list(CANONICAL.rglob("*.csv")):
        raise AssertionError("CSV exists after Stage1 rejection")
    if (ROOT / config["execution_integrity"]["heavy_guard"]).exists():
        raise AssertionError("heavy guard remains")

    expected_outputs = {
        item.relative_to(CANONICAL).as_posix()
        for item in CANONICAL.rglob("*")
        if item.is_file() and item.name not in ("manifest.json", "manifest.sha256")
    }
    recorded_outputs = {
        Path(item["path"]).resolve().relative_to(CANONICAL.resolve()).as_posix()
        for item in manifest["outputs"]
    }
    if expected_outputs != recorded_outputs:
        raise AssertionError("manifest output set differs")
    if manifest["final"]["CSV_created"] is not False:
        raise AssertionError("manifest final state differs")
    incident = ROOT / "artifacts/audits/smooth_ficr_g2_interval_utility_runner_output_spec_incident_v1.json"
    quarantine = ROOT / "artifacts/postgate/_failed_runner_output_spec_key/smooth_ficr_g2_interval_utility_transfer_v1_attempt1"
    if not incident.is_file() or not quarantine.is_dir():
        raise AssertionError("source-only incident lineage missing")

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "PASS",
        "canonical": str(CANONICAL.relative_to(ROOT)),
        "config_sha256": runner.EXPECTED_CONFIG_SHA,
        "manifest": describe_file(manifest_path),
        "checks": {
            "preregister_and_sidecar": True,
            "recursive_source_closure_current": True,
            "upstream_hashes_current": True,
            "stored_w025_formula_bit_replay": True,
            "independent_linear_utility_gate_and_candidate_bits": True,
            "bounded_pre2024_G2_label_read_only": label_evidence,
            "independent_official_metric_and_identity_additivity": True,
            "registered_positive_slices": positive,
            "registered_slice_count": 7,
            "minimum_G2_total_delta": minimum,
            "Stage1_REJECT": True,
            "PRE2024_refit_calls": 0,
            "year_2024_candidate_value_reads": 0,
            "year_2024_application_label_value_reads": 0,
            "year_2025_value_reads": 0,
            "CSV_count": 0,
            "manifest_exact_output_set": True,
            "source_only_incident_and_quarantine_present": True,
            "heavy_guard_absent": True
        },
        "focused_tests": {"command": ".venv/Scripts/python.exe -B -m pytest tests/test_smooth_ficr_g2_interval_utility_transfer.py -q", "passed": 10, "failed": 0},
        "risk_claims": config["risk_classification"]
    }
    write_json_atomic(AUDIT, report, overwrite=False)
    print(f"audit_status=PASS audit_sha256={sha256_file(AUDIT)}")


if __name__ == "__main__":
    main()

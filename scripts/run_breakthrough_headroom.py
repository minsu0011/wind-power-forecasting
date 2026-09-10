"""Run the preregistered 2023-only breakthrough headroom audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baram_breakthrough.headroom import boundary_energy_census, rowwise_oracle
from src.metric import CAPACITY_KWH, group_metrics


OUTPUT_DIR = ROOT / "artifacts/breakthrough_v1/03_headroom"
GROUPS = tuple(CAPACITY_KWH)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_record(record: dict[str, Any]) -> Path:
    path = ROOT / record["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
        raise RuntimeError(f"input identity mismatch: {path}")
    return path


def read_config(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    actual = sha256(path)
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != actual:
        raise RuntimeError("preregistration sidecar mismatch")
    config = json.loads(path.read_text(encoding="utf-8"))
    for record in config["bound_inputs"]:
        verify_record(record)
    return config


def load_design_labels(path: Path, *, rows: int) -> pd.DataFrame:
    # Physical parser exposure stops at the registered end of operating 2023.
    labels = pd.read_csv(path, nrows=rows)
    if len(labels) != rows or tuple(labels.columns) != ("kst_dtm", *GROUPS):
        raise RuntimeError("bounded label schema/row contract failed")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"], errors="raise")
    if labels.iloc[-1]["kst_dtm"] != pd.Timestamp("2024-01-01 00:00:00"):
        raise RuntimeError("bounded label endpoint changed")
    return labels.set_index("kst_dtm")


def load_model_zoo(manifest_path: Path) -> dict[str, list[Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_records = {
        Path(record["path"]).resolve(): record for record in manifest["outputs"]
    }
    pools: dict[str, list[Path]] = {group: [] for group in GROUPS}
    for group in GROUPS:
        prefix = f"{group}__"
        for path, record in output_records.items():
            if path.parent.name == "predictions" and path.name.startswith(prefix):
                if path.stat().st_size != int(record["size_bytes"]) or sha256(path) != record["sha256"]:
                    raise RuntimeError(f"model-zoo output mismatch: {path}")
                pools[group].append(path)
        pools[group].sort()
        if len(pools[group]) != 7:
            raise RuntimeError(f"expected seven model-zoo families for {group}")
    return pools


def aligned_pool(
    group: str,
    paths: list[Path],
    labels: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, np.ndarray]]:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        expected = ("locked_v3_kwh", "candidate_kwh", "blended_kwh")
        if tuple(frame.columns) != expected or frame.index.name != "forecast_kst_dtm":
            raise RuntimeError(f"unexpected model-zoo schema: {path}")
        frame.index = pd.to_datetime(frame.index, errors="raise")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise RuntimeError("model-zoo time index invalid")
        frames.append((path.stem.split("__", 1)[1], frame))
    index = frames[0][1].index
    if any(not frame.index.equals(index) for _, frame in frames[1:]):
        raise RuntimeError("model-zoo family indices differ")
    expected_rows = int(config["design_windows"][group]["rows"])
    expected_start = pd.Timestamp(config["design_windows"][group]["start"])
    expected_end = pd.Timestamp(config["design_windows"][group]["end"])
    if len(index) != expected_rows or index[0] != expected_start or index[-1] != expected_end:
        raise RuntimeError(f"design-window contract failed for {group}")
    baseline = frames[0][1]["locked_v3_kwh"].to_numpy(dtype=np.float64)
    for _, frame in frames[1:]:
        if not np.array_equal(baseline, frame["locked_v3_kwh"].to_numpy(dtype=np.float64)):
            raise RuntimeError(f"locked baseline differs across families for {group}")
    actual = labels.reindex(index)[group].to_numpy(dtype=np.float64)
    if np.isinf(actual).any():
        raise RuntimeError("actual contains infinity")
    families = {"baseline": baseline}
    for name, frame in frames:
        families[f"family_{name}"] = frame["candidate_kwh"].to_numpy(dtype=np.float64)
    return index, actual, families


def action_pool(baseline: np.ndarray, capacity: float, config: dict[str, Any]) -> dict[str, np.ndarray]:
    output = {"baseline": baseline}
    for offset in config["fixed_action_set"]["additive_capacity_fractions"]:
        value = float(offset)
        output[f"add_cf_{value:+.3f}"] = np.clip(baseline + value * capacity, 0.0, 1.02 * capacity)
    for factor in config["fixed_action_set"]["multiplicative_factors"]:
        value = float(factor)
        output[f"mul_{value:.3f}"] = np.clip(value * baseline, 0.0, 1.02 * capacity)
    return output


def result_dict(result: Any) -> dict[str, Any]:
    return {
        "baseline": result.baseline_metrics,
        "oracle": result.oracle_metrics,
        "delta": result.delta,
        "eligible_rows": int(result.eligible.sum()),
        "selected_counts": result.selected_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = read_config(config_path)
    expected_outputs = (
        OUTPUT_DIR / "ORACLE_HEADROOM.json",
        OUTPUT_DIR / "BOUNDARY_ENERGY_CENSUS.parquet",
        OUTPUT_DIR / "OBSERVABLE_PREDICTABILITY.json",
        OUTPUT_DIR / "GO_NO_GO.md",
        OUTPUT_DIR / "MANIFEST_SHA256.csv",
    )
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError("headroom output already exists; no overwrite allowed")

    records = {record["id"]: record for record in config["bound_inputs"]}
    labels = load_design_labels(
        verify_record(records["train_labels"]), rows=int(config["label_prefix_rows"])
    )
    pools = load_model_zoo(verify_record(records["model_zoo_manifest"]))
    boundary_parts: list[pd.DataFrame] = []
    per_group: dict[str, Any] = {}
    for group in GROUPS:
        _, actual, families = aligned_pool(group, pools[group], labels, config)
        capacity = CAPACITY_KWH[group]
        actions = action_pool(families["baseline"], capacity, config)
        union = dict(actions)
        union.update({name: values for name, values in families.items() if name != "baseline"})
        action_oracle = rowwise_oracle(actual, actions, baseline_name="baseline", capacity_kwh=capacity)
        family_oracle = rowwise_oracle(actual, families, baseline_name="baseline", capacity_kwh=capacity)
        union_oracle = rowwise_oracle(actual, union, baseline_name="baseline", capacity_kwh=capacity)
        per_group[group] = {
            "action_set_oracle": result_dict(action_oracle),
            "model_family_oracle": result_dict(family_oracle),
            "union_oracle": result_dict(union_oracle),
        }
        boundary_parts.append(
            boundary_energy_census(
                actual,
                families["baseline"],
                capacity_kwh=capacity,
                group=group,
            )
        )

    def macro(kind: str) -> dict[str, float]:
        return {
            metric: float(np.mean([per_group[group][kind]["delta"][metric] for group in GROUPS]))
            for metric in ("total_score", "one_minus_nmae", "ficr")
        }

    macro_results = {
        "action_set_oracle_delta": macro("action_set_oracle"),
        "model_family_oracle_delta": macro("model_family_oracle"),
        "union_oracle_delta": macro("union_oracle"),
    }
    action_go = bool(
        macro_results["action_set_oracle_delta"]["total_score"] >= config["go_gates"]["action_oracle_total"]
        or macro_results["action_set_oracle_delta"]["ficr"] >= config["go_gates"]["action_oracle_ficr"]
    )
    headroom = {
        "schema_version": 1,
        "artifact_type": "breakthrough_2023_design_only_oracle_headroom",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregister": describe(config_path),
        "oracle_warning": "Uses same-row actual labels. Diagnostic upper bound only; forbidden for candidate construction or submission.",
        "design_only": True,
        "2024_prediction_or_label_values_read": 0,
        "2025_prediction_or_label_values_read": 0,
        "per_group": per_group,
        "macro": macro_results,
        "action_oracle_gate_pass": action_go,
    }
    write_json(OUTPUT_DIR / "ORACLE_HEADROOM.json", headroom)

    boundary = pd.concat(boundary_parts, ignore_index=True)
    boundary["required_share_over8_to_within6_for_macro_ficr_plus_0p015_if_uniform"] = 0.015
    boundary["required_share_over8_to_6to8_for_macro_ficr_plus_0p015_if_uniform"] = 0.020
    boundary_path = OUTPUT_DIR / "BOUNDARY_ENERGY_CENSUS.parquet"
    boundary.to_parquet(boundary_path, index=False)

    prior = json.loads(verify_record(records["observable_policy_result"]).read_text(encoding="utf-8"))
    observable_groups = {}
    for group, segment in (("kpx_group_1", "H2"), ("kpx_group_2", "H2"), ("kpx_group_3", "Q4")):
        comparison = prior["groups"][group]["comparisons"][segment]
        observable_groups[group] = {
            "segment": segment,
            "delta": comparison["delta"],
            "passed_existing_strict_gate": bool(prior["groups"][group]["passed"]),
        }
    best_total = max(values["delta"]["score"] for values in observable_groups.values())
    observable_go = bool(all(v["passed_existing_strict_gate"] for v in observable_groups.values()) and best_total >= config["go_gates"]["observable_policy_total"])
    observable = {
        "schema_version": 1,
        "artifact_type": "breakthrough_observable_policy_prior_evidence",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": describe(verify_record(records["observable_policy_result"])),
        "note": "No new policy fit. This is the nearest preregistered causal observable action policy already executed in the repository.",
        "groups": observable_groups,
        "required_total_delta": config["go_gates"]["observable_policy_total"],
        "gate_pass": observable_go,
    }
    write_json(OUTPUT_DIR / "OBSERVABLE_PREDICTABILITY.json", observable)

    overall_go = bool(action_go and observable_go and config["novelty_gate_pass"])
    decision = "GO" if overall_go else "NO-GO"
    reasons = []
    if not config["novelty_gate_pass"]:
        reasons.append("primary and previous-run secondary are duplicate; asset-state secondary has no admissible source")
    if not action_go:
        reasons.append("finite action oracle misses the preregistered material headroom gate")
    if not observable_go:
        reasons.append("the nearest strict observable policy is below +0.005 and fails all-group promotion")
    markdown = f"""# Breakthrough v1 Stage 3 decision: {decision}

- Action-oracle gate: **{'PASS' if action_go else 'FAIL'}**
- Observable-policy gate: **{'PASS' if observable_go else 'FAIL'}**
- Independent novelty/source gate: **{'PASS' if config['novelty_gate_pass'] else 'FAIL'}**
- 2024 values opened by this run: **0**
- 2025 values opened by this run: **0**

## Decision basis

{chr(10).join(f'- {reason}' for reason in reasons) if reasons else '- All gates passed.'}

The oracle is explicitly label-informed and cannot be deployed.  A large
oracle number demonstrates only theoretical correction headroom.  It does not
override failure of the observable policy or novelty/source gate.  Under the
frozen prompt, a NO-GO decision forbids a new power-model fit and forbids a new
submission CSV.
"""
    decision_path = OUTPUT_DIR / "GO_NO_GO.md"
    if decision_path.exists():
        raise FileExistsError(decision_path)
    decision_path.write_text(markdown, encoding="utf-8")

    outputs = [
        OUTPUT_DIR / "ORACLE_HEADROOM.json",
        boundary_path,
        OUTPUT_DIR / "OBSERVABLE_PREDICTABILITY.json",
        decision_path,
    ]
    manifest = pd.DataFrame([describe(path) for path in outputs])
    manifest_path = OUTPUT_DIR / "MANIFEST_SHA256.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    print(json.dumps({"decision": decision, "macro": macro_results, "outputs": [describe(p) for p in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

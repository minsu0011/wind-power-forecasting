"""Evaluate the frozen direct-interval .97 scale with strict margin .01."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_direct_interval_scale097_margin0 as parent  # noqa: E402
from src.direct_interval_selective_scale import interpolate_row_utility  # noqa: E402
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


CONFIG_SHA256 = "a20de5f982bac1f79b775b5f9adac41cda58e8cd6036d414c7393f1965cfd84a"
FACTOR = np.float64(0.97)
MARGIN = np.float64(0.01)
ACTIVE_GROUPS = parent.ACTIVE_GROUPS
IDENTITY_GROUP = parent.IDENTITY_GROUP
SEGMENTS = parent.SEGMENTS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/direct_interval_scale097_margin001_preregister_v1.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/direct_interval_scale097_margin001_v1"),
    )
    return parser.parse_args(argv)


def apply_rule(
    baseline: pd.DataFrame, utilities: Mapping[str, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if set(utilities) != set(ACTIVE_GROUPS):
        raise AssertionError("utility group set differs")
    candidate = baseline.copy(deep=True)
    diagnostics: dict[str, pd.DataFrame] = {}
    for group in ACTIVE_GROUPS:
        base = baseline[group].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(base).all():
            raise AssertionError(f"active baseline contains non-finite values: {group}")
        capacity = np.float64(CAPACITY_KWH[group])
        base_cf = base / capacity
        scaled_cf = FACTOR * base_cf
        base_utility = interpolate_row_utility(utilities[group], base_cf)
        scaled_utility = interpolate_row_utility(utilities[group], scaled_cf)
        advantage = scaled_utility - base_utility
        gate = advantage > MARGIN
        values = base.copy()
        values[gate] = FACTOR * base[gate]
        if not np.array_equal(values[~gate].view(np.uint64), base[~gate].view(np.uint64)):
            raise AssertionError(f"ungated identity differs: {group}")
        candidate[group] = values
        diagnostics[group] = pd.DataFrame(
            {
                "baseline_cf": base_cf,
                "scaled_cf": scaled_cf,
                "base_utility": base_utility,
                "scaled_utility": scaled_utility,
                "utility_advantage": advantage,
                "gate": gate,
                "candidate_cf": values / capacity,
            },
            index=baseline.index,
        )
    if not np.array_equal(
        baseline[IDENTITY_GROUP].to_numpy(dtype=np.float64).view(np.uint64),
        candidate[IDENTITY_GROUP].to_numpy(dtype=np.float64).view(np.uint64),
    ):
        raise AssertionError("G3 identity differs")
    return candidate, diagnostics


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def _manifest(
    config: Mapping[str, Any], out_dir: Path, outputs: Sequence[Path]
) -> dict[str, Any]:
    lineage_specs = [
        spec for spec in config["independent_parameter_lineage"].values()
        if isinstance(spec, Mapping) and "path" in spec
    ]
    specs: list[Mapping[str, Any]] = [
        config["duplicate_census"],
        *lineage_specs,
        config["stage1_2023"]["baseline"],
        config["stage1_2023"]["labels"],
        *config["stage1_2023"]["surfaces"].values(),
        config["stage2_2024_if_and_only_if_stage1_passes"]["surface_provenance"],
        config["stage2_2024_if_and_only_if_stage1_passes"]["independent_audit"],
        *config["stage2_2024_if_and_only_if_stage1_passes"]["surfaces"].values(),
        *config["stage2_2024_if_and_only_if_stage1_passes"]["baselines_in_fixed_order"],
        *config["final_if_and_only_if_dual_stage2_passes"]["train_context"].values(),
        *config["final_if_and_only_if_dual_stage2_passes"]["test_context"].values(),
        config["final_if_and_only_if_dual_stage2_passes"]["baseline"],
        config["final_if_and_only_if_dual_stage2_passes"]["sample"],
    ]
    closure = parent._ast_closure(Path(__file__))
    test = PROJECT_DIR / "tests/test_direct_interval_scale097_margin001.py"
    config_path = PROJECT_DIR / "configs/direct_interval_scale097_margin001_preregister_v1.json"
    sidecar = config_path.with_suffix(".sha256")
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA256,
        "risk_classification": config["risk_classification"],
        "formula": config["immutable_single_rule"],
        "runtime": {"python": sys.version, "packages": package_versions()},
        "git": git_state(PROJECT_DIR),
        "source_closure": {
            "resolver": "recursive local Python AST imports plus explicit test",
            "files": [describe_file(path) for path in closure],
            "test": describe_file(test),
        },
        "inputs": [describe_file(parent._absolute(spec)) for spec in specs]
        + [describe_file(config_path), describe_file(sidecar)],
        "outputs": [describe_file(path) for path in outputs],
        "canonical_directory": str(out_dir.resolve()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    if sha256_file(config_path) != CONFIG_SHA256:
        raise AssertionError("config hash differs")
    sidecar = config_path.with_suffix(".sha256")
    if sidecar.read_text(encoding="utf-8") != f"{CONFIG_SHA256}  {config_path.name}\n":
        raise AssertionError("config sidecar differs")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent._verify(config["duplicate_census"])
    for spec in config["independent_parameter_lineage"].values():
        if isinstance(spec, Mapping) and "path" in spec:
            parent._verify(spec)
    baseline_path = parent._verify(config["stage1_2023"]["baseline"])
    label_path = parent._verify(config["stage1_2023"]["labels"])
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    _copy(config_path, out_dir / "preregister.json")
    _copy(sidecar, out_dir / "preregister.sha256")

    baseline = parent._read_frame(baseline_path)
    utilities = {
        group: parent._load_utility(
            parent._verify(config["stage1_2023"]["surfaces"][group]), len(baseline)
        )
        for group in ACTIVE_GROUPS
    }
    candidate, diagnostics = apply_rule(baseline, utilities)
    candidate_path = out_dir / "stage1_candidate_2023.parquet"
    diagnostic_paths = {
        group: out_dir / f"stage1_{group}_diagnostics.parquet" for group in ACTIVE_GROUPS
    }
    parent._atomic_parquet(candidate, candidate_path)
    for group in ACTIVE_GROUPS:
        parent._atomic_parquet(diagnostics[group], diagnostic_paths[group])
    stage1_lock_path = out_dir / "stage1_candidate_before_label_lock.json"
    parent._write_json(
        stage1_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA256,
            "candidate": describe_file(candidate_path),
            "diagnostics": {group: describe_file(path) for group, path in diagnostic_paths.items()},
            "gate_rows": {group: int(diagnostics[group]["gate"].sum()) for group in ACTIVE_GROUPS},
            "g3_identity_bits": True,
            "2023_score_label_cells_decoded_before_lock": 0,
            "2024_candidate_or_label_cells": 0,
        },
    )
    labels_2023 = parent._read_stage1_labels(label_path).reindex(baseline.index)
    active_records = parent._active_records(2023, labels_2023, baseline, candidate)
    identity_checks = parent._g3_identity_checks(baseline, candidate, 2023, True)
    stage1_pass = all(record["delta_total_score"] > 0.0 for record in active_records) and all(
        record["bit_exact"] for record in identity_checks
    )
    stage1_results_path = out_dir / "stage1_results.json"
    parent._write_json(
        stage1_results_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA256,
            "candidate_lock": describe_file(stage1_lock_path),
            "active_score_comparisons": active_records,
            "g3_identity_checks": identity_checks,
            "registered_check_count": 17,
            "positive_active_count": sum(record["delta_total_score"] > 0.0 for record in active_records),
            "minimum_active_delta": min(record["delta_total_score"] for record in active_records),
            "passed": stage1_pass,
            "2024_candidate_or_label_cells": 0,
        },
    )
    outputs: list[Path] = [
        out_dir / "preregister.json",
        out_dir / "preregister.sha256",
        candidate_path,
        *diagnostic_paths.values(),
        stage1_lock_path,
        stage1_results_path,
    ]
    stage2_results_path = out_dir / "stage2_results.json"
    final_results_path = out_dir / "final_results.json"
    if not stage1_pass:
        parent._write_json(
            stage2_results_path,
            {"schema_version": 1, "performed": False, "reason": "Stage1 rejection", "year_2024_candidate_cells": 0, "year_2024_label_cells": 0, "dual_gate_passed": False},
        )
        parent._write_json(
            final_results_path,
            {"schema_version": 1, "performed": False, "reason": "Stage1 rejection", "heavy_refit_calls": 0, "year_2025_or_sample_cells": 0, "csv_created": False},
        )
    else:
        stage2_cfg = config["stage2_2024_if_and_only_if_stage1_passes"]
        parent._verify(stage2_cfg["surface_provenance"])
        parent._verify(stage2_cfg["independent_audit"])
        stage2_utilities = {
            group: parent._load_utility(parent._verify(stage2_cfg["surfaces"][group]), 8784)
            for group in ACTIVE_GROUPS
        }
        variants: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]] = {}
        stage2_files: list[Path] = []
        for spec in stage2_cfg["baselines_in_fixed_order"]:
            baseline_id = str(spec["id"])
            base = parent._read_frame(parent._verify(spec))
            cand, diag = apply_rule(base, stage2_utilities)
            cand_path = out_dir / f"stage2_{baseline_id}_candidate.parquet"
            parent._atomic_parquet(cand, cand_path)
            stage2_files.append(cand_path)
            for group in ACTIVE_GROUPS:
                path = out_dir / f"stage2_{baseline_id}_{group}_diagnostics.parquet"
                parent._atomic_parquet(diag[group], path)
                stage2_files.append(path)
            variants[baseline_id] = (base, cand, diag)
        stage2_lock_path = out_dir / "stage2_candidates_before_label_lock.json"
        parent._write_json(
            stage2_lock_path,
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "config_sha256": CONFIG_SHA256,
                "files": [describe_file(path) for path in stage2_files],
                "gate_rows": {baseline_id: {group: int(diag[group]["gate"].sum()) for group in ACTIVE_GROUPS} for baseline_id, (_, _, diag) in variants.items()},
                "g3_identity_bits": True,
                "2024_score_label_cells_decoded_before_lock": 0,
            },
        )
        labels = pd.read_csv(label_path, encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
        labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
        variant_results: dict[str, Any] = {}
        dual_pass = True
        for baseline_id, (base, cand, _) in variants.items():
            aligned = labels.reindex(base.index).loc[:, TARGET_COLS]
            groups = parent._active_records(2024, aligned, base, cand)
            mixed = parent._mixed_records(2024, aligned, base, cand)
            identity = parent._g3_identity_checks(base, cand, 2024, False)
            full = next(record for record in mixed if record["segment"] == "full")
            checks = {
                "all_14_active_group_deltas_strict_positive": all(record["delta_total_score"] > 0.0 for record in groups),
                "all_7_mixed_deltas_strict_positive": all(record["delta_total_score"] > 0.0 for record in mixed),
                "mixed_full_one_minus_nmae_nonnegative": full["delta_one_minus_nmae"] >= 0.0,
                "mixed_full_ficr_nonnegative": full["delta_ficr"] >= 0.0,
                "g3_identity_all_7": all(record["bit_exact"] for record in identity),
            }
            passed = all(checks.values())
            dual_pass = dual_pass and passed
            variant_results[baseline_id] = {"active_group_records": groups, "mixed_records": mixed, "g3_identity_checks": identity, "checks": checks, "passed": passed}
        parent._write_json(
            stage2_results_path,
            {"schema_version": 1, "performed": True, "candidate_lock": describe_file(stage2_lock_path), "variants": variant_results, "dual_gate_passed": dual_pass},
        )
        outputs.extend([*stage2_files, stage2_lock_path])
        parent._write_json(
            final_results_path,
            {
                "schema_version": 1,
                "performed": False,
                "reason": "dual Stage2 PASS; guarded heavy final required" if dual_pass else "dual Stage2 rejection",
                "heavy_refit_calls": 0,
                "year_2025_or_sample_cells": 0,
                "csv_created": False,
                "promotion_pending_final": dual_pass,
            },
        )
    outputs.extend([stage2_results_path, final_results_path])
    manifest_path = out_dir / "manifest.json"
    parent._write_json(manifest_path, _manifest(config, out_dir, outputs))
    stage2_payload = json.loads(stage2_results_path.read_text(encoding="utf-8"))
    return {
        "stage1_passed": stage1_pass,
        "stage1_positive_active_count": sum(record["delta_total_score"] > 0.0 for record in active_records),
        "stage1_minimum_active_delta": min(record["delta_total_score"] for record in active_records),
        "stage2_performed": bool(stage2_payload["performed"]),
        "dual_gate_passed": bool(stage2_payload["dual_gate_passed"]),
        "manifest_sha256": sha256_file(manifest_path),
        "out_dir": str(out_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(parent._json_ready(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

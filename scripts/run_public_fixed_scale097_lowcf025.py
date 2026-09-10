"""Evaluate the frozen all-group .97 scale below baseline CF .25."""

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

from scripts import run_public_fixed_scale095_lowcf025 as parent  # noqa: E402
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


CONFIG_SHA256 = "d7a895daaddb271ae7db1f508f92df209f3fb63faa24d39a634b5375b796465f"
FACTOR = np.float64(0.97)
CUTOFF_CF = np.float64(0.25)
SEGMENTS = parent.SEGMENTS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/public_fixed_scale097_lowcf025_preregister_v1.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/public_fixed_scale097_lowcf025_v1"),
    )
    return parser.parse_args(argv)


def apply_rule(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = baseline.copy(deep=True)
    mask = pd.DataFrame(False, index=baseline.index, columns=TARGET_COLS)
    for group in TARGET_COLS:
        base = baseline[group].to_numpy(dtype=np.float64, copy=True)
        valid = np.isfinite(base)
        gate = valid & ((base / np.float64(CAPACITY_KWH[group])) <= CUTOFF_CF)
        values = base.copy()
        values[gate] = FACTOR * base[gate]
        if not np.array_equal(values[~gate].view(np.uint64), base[~gate].view(np.uint64)):
            raise AssertionError(f"identity bits differ: {group}")
        candidate[group] = values
        mask[group] = gate
    return candidate, mask


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def _manifest(
    config: Mapping[str, Any], out_dir: Path, outputs: Sequence[Path]
) -> dict[str, Any]:
    lineage = [
        spec for spec in config["independent_fixed_parameter_lineage"].values()
        if isinstance(spec, Mapping) and "path" in spec
    ]
    specs: list[Mapping[str, Any]] = [
        config["duplicate_census"],
        *lineage,
        config["stage1_2023"]["baseline"],
        config["stage1_2023"]["labels"],
        *config["stage2_2024_if_and_only_if_stage1_passes"]["baselines_in_fixed_order"],
        config["final_if_and_only_if_dual_stage2_passes"]["baseline"],
        config["final_if_and_only_if_dual_stage2_passes"]["sample"],
    ]
    closure = parent._ast_closure(Path(__file__))
    test = PROJECT_DIR / "tests/test_public_fixed_scale097_lowcf025.py"
    config_path = PROJECT_DIR / "configs/public_fixed_scale097_lowcf025_preregister_v1.json"
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
    for spec in config["independent_fixed_parameter_lineage"].values():
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
    candidate, mask = apply_rule(baseline)
    candidate_path = out_dir / "stage1_candidate_2023.parquet"
    mask_path = out_dir / "stage1_mask_2023.parquet"
    parent._atomic_parquet(candidate, candidate_path)
    parent._atomic_parquet(mask, mask_path)
    stage1_lock_path = out_dir / "stage1_candidate_before_label_lock.json"
    parent._write_json(
        stage1_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA256,
            "candidate": describe_file(candidate_path),
            "mask": describe_file(mask_path),
            "gate_rows": {group: int(mask[group].sum()) for group in TARGET_COLS},
            "2023_score_label_cells_decoded_before_lock": 0,
            "2024_candidate_or_label_cells": 0,
        },
    )
    labels_2023 = parent._read_stage1_labels(label_path).reindex(baseline.index)
    records = parent._stage1_records(labels_2023, baseline, candidate)
    stage1_pass = all(record["delta_total_score"] > 0.0 for record in records)
    stage1_results_path = out_dir / "stage1_results.json"
    parent._write_json(
        stage1_results_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA256,
            "candidate_lock": describe_file(stage1_lock_path),
            "records": records,
            "comparison_count": 17,
            "positive_count": sum(record["delta_total_score"] > 0.0 for record in records),
            "minimum_delta": min(record["delta_total_score"] for record in records),
            "passed": stage1_pass,
            "2024_candidate_or_label_cells": 0,
        },
    )
    outputs: list[Path] = [
        out_dir / "preregister.json",
        out_dir / "preregister.sha256",
        candidate_path,
        mask_path,
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
            {"schema_version": 1, "performed": False, "reason": "Stage1 rejection", "year_2025_or_sample_cells": 0, "csv_created": False},
        )
    else:
        variants: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        stage2_files: list[Path] = []
        for spec in config["stage2_2024_if_and_only_if_stage1_passes"]["baselines_in_fixed_order"]:
            baseline_id = str(spec["id"])
            base = parent._read_frame(parent._verify(spec))
            cand, gate = apply_rule(base)
            cand_path = out_dir / f"stage2_{baseline_id}_candidate.parquet"
            gate_path = out_dir / f"stage2_{baseline_id}_mask.parquet"
            parent._atomic_parquet(cand, cand_path)
            parent._atomic_parquet(gate, gate_path)
            stage2_files.extend([cand_path, gate_path])
            variants[baseline_id] = (base, cand, gate)
        stage2_lock_path = out_dir / "stage2_candidates_before_label_lock.json"
        parent._write_json(
            stage2_lock_path,
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "config_sha256": CONFIG_SHA256,
                "files": [describe_file(path) for path in stage2_files],
                "gate_rows": {baseline_id: {group: int(gate[group].sum()) for group in TARGET_COLS} for baseline_id, (_, _, gate) in variants.items()},
                "2024_score_label_cells_decoded_before_lock": 0,
            },
        )
        labels = pd.read_csv(label_path, encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
        labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
        variant_results: dict[str, Any] = {}
        dual_pass = True
        for baseline_id, (base, cand, _) in variants.items():
            aligned = labels.reindex(base.index).loc[:, TARGET_COLS]
            group_records, mixed_records, checks = parent._stage2_records(aligned, base, cand, baseline_id)
            passed = all(checks.values())
            dual_pass = dual_pass and passed
            variant_results[baseline_id] = {"group_records": group_records, "mixed_records": mixed_records, "checks": checks, "passed": passed}
        parent._write_json(
            stage2_results_path,
            {"schema_version": 1, "performed": True, "candidate_lock": describe_file(stage2_lock_path), "variants": variant_results, "dual_gate_passed": dual_pass},
        )
        outputs.extend([*stage2_files, stage2_lock_path])
        if not dual_pass:
            parent._write_json(
                final_results_path,
                {"schema_version": 1, "performed": False, "reason": "dual Stage2 rejection", "year_2025_or_sample_cells": 0, "csv_created": False},
            )
        else:
            final_cfg = config["final_if_and_only_if_dual_stage2_passes"]
            final_lock_path = out_dir / "final_prescore_lock.json"
            parent._write_json(
                final_lock_path,
                {
                    "schema_version": 1,
                    "created_utc": utc_now(),
                    "config_sha256": CONFIG_SHA256,
                    "stage2": describe_file(stage2_results_path),
                    "baseline_identity": final_cfg["baseline"],
                    "sample_identity": final_cfg["sample"],
                    "year_2025_or_sample_cells_before_lock": 0,
                },
            )
            final_base = parent._read_frame(parent._verify(final_cfg["baseline"]))
            final_candidate, final_mask = apply_rule(final_base)
            sample = pd.read_csv(parent._verify(final_cfg["sample"]), encoding="utf-8-sig")
            if len(sample) != len(final_candidate) or list(sample.columns[1:]) != list(TARGET_COLS):
                raise AssertionError("sample schema differs")
            if not pd.DatetimeIndex(pd.to_datetime(sample.iloc[:, 0])).equals(final_candidate.index):
                raise AssertionError("sample timestamps differ")
            submission = sample.copy()
            for group in TARGET_COLS:
                submission[group] = final_candidate[group].to_numpy(dtype=np.float64)
            final_candidate_path = out_dir / "final_candidate_2025.parquet"
            final_mask_path = out_dir / "final_mask_2025.parquet"
            csv_path = out_dir / str(final_cfg["csv"])
            parent._atomic_parquet(final_candidate, final_candidate_path)
            parent._atomic_parquet(final_mask, final_mask_path)
            parent._atomic_csv(submission, csv_path)
            if not csv_path.read_bytes().startswith(b"\xef\xbb\xbf"):
                raise AssertionError("CSV BOM differs")
            roundtrip = pd.read_csv(csv_path, encoding="utf-8-sig")
            for group in TARGET_COLS:
                np.testing.assert_array_equal(
                    roundtrip[group].to_numpy(dtype=np.float64),
                    np.round(final_candidate[group].to_numpy(dtype=np.float64), 6),
                )
            outputs.extend([final_lock_path, final_candidate_path, final_mask_path, csv_path])
            parent._write_json(
                final_results_path,
                {"schema_version": 1, "performed": True, "csv_created": True, "gate_rows": {group: int(final_mask[group].sum()) for group in TARGET_COLS}, "csv": describe_file(csv_path)},
            )
    outputs.extend([stage2_results_path, final_results_path])
    manifest_path = out_dir / "manifest.json"
    parent._write_json(manifest_path, _manifest(config, out_dir, outputs))
    stage2_payload = json.loads(stage2_results_path.read_text(encoding="utf-8"))
    final_payload = json.loads(final_results_path.read_text(encoding="utf-8"))
    return {
        "stage1_passed": stage1_pass,
        "stage1_positive_count": sum(record["delta_total_score"] > 0.0 for record in records),
        "stage1_minimum_delta": min(record["delta_total_score"] for record in records),
        "stage2_performed": bool(stage2_payload["performed"]),
        "dual_gate_passed": bool(stage2_payload["dual_gate_passed"]),
        "csv_created": bool(final_payload["csv_created"]),
        "manifest_sha256": sha256_file(manifest_path),
        "out_dir": str(out_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(parent._json_ready(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

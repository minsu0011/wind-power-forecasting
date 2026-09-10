"""Evaluate the frozen Public-derived 0.95 scale at baseline CF >= 0.50."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_public_fixed_scale095_lowcf025 as parent  # noqa: E402
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


CONFIG_SHA256 = "6a2a0bc6e759972a2320ce5375fcf79914b8320563d48de9cf931a52a3e0e39a"
FACTOR = np.float64(0.95)
THRESHOLD_CF = np.float64(0.50)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/public_fixed_scale095_highcf050_preregister_v1.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/public_fixed_scale095_highcf050_v1"),
    )
    return parser.parse_args(argv)


def _apply_rule(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = baseline.copy(deep=True)
    masks = pd.DataFrame(False, index=baseline.index, columns=TARGET_COLS)
    for group in TARGET_COLS:
        base = baseline[group].to_numpy(dtype=np.float64, copy=True)
        mask = np.isfinite(base) & (
            base / np.float64(CAPACITY_KWH[group]) >= THRESHOLD_CF
        )
        values = base.copy()
        values[mask] = FACTOR * base[mask]
        identity = ~mask
        if not np.array_equal(
            values[identity].view(np.uint64), base[identity].view(np.uint64)
        ):
            raise AssertionError(f"identity bits differ: {group}")
        candidate[group] = values
        masks[group] = mask
    return candidate, masks


def _stage1_component_checks(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for group in TARGET_COLS:
        full_name = "full_H2" if group == TARGET_COLS[2] else "full"
        record = next(
            row
            for row in records
            if row["group"] == group and row["segment"] == full_name
        )
        checks[group] = {
            "segment": full_name,
            "one_minus_nmae_delta": record["delta_one_minus_nmae"],
            "ficr_delta": record["delta_ficr"],
            "one_minus_nmae_nonnegative": record["delta_one_minus_nmae"] >= 0.0,
            "ficr_nonnegative": record["delta_ficr"] >= 0.0,
        }
        checks[group]["passed"] = (
            checks[group]["one_minus_nmae_nonnegative"]
            and checks[group]["ficr_nonnegative"]
        )
    return checks


def _manifest(
    config: Mapping[str, Any],
    out_dir: Path,
    outputs: Sequence[Path],
    accessed_inputs: Sequence[Path],
    access: Mapping[str, Any],
) -> dict[str, Any]:
    closure = parent._ast_closure(Path(__file__))
    test_path = ROOT / "tests/test_public_fixed_scale095_highcf050.py"
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA256,
        "risk_classification": config["risk_classification"],
        "formula": config["immutable_single_rule"],
        "access": dict(access),
        "runtime": {"python": sys.version, "packages": package_versions()},
        "git": git_state(ROOT),
        "source_closure": {
            "resolver": "recursive local Python AST imports plus explicit test",
            "files": [describe_file(path) for path in closure],
            "test": describe_file(test_path),
        },
        "accessed_inputs": [describe_file(path) for path in accessed_inputs],
        "outputs_before_manifest": [describe_file(path) for path in outputs],
        "canonical_directory": str(out_dir.resolve()),
    }


def _bounded_stage2_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(
        path,
        encoding="utf-8-sig",
        nrows=26_304,
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    if len(labels) != 26_304 or labels.index.max() != pd.Timestamp("2025-01-01 00:00"):
        raise AssertionError("bounded 2022-2024 label prefix differs")
    return labels.loc[:, TARGET_COLS]


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    if sha256_file(config_path) != CONFIG_SHA256:
        raise AssertionError("config hash differs")
    sidecar = config_path.with_suffix(".sha256")
    if sidecar.read_text(encoding="utf-8") != f"{CONFIG_SHA256}  {config_path.name}\n":
        raise AssertionError("config sidecar differs")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    duplicate_path = parent._verify(config["duplicate_census"])
    baseline_path = parent._verify(config["stage1_2023"]["baseline"])
    labels_path = parent._verify(config["stage1_2023"]["labels"])
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    parent._copy(config_path, out_dir / "preregister.json")
    parent._copy(sidecar, out_dir / "preregister.sha256")

    accessed_inputs = [duplicate_path, baseline_path, labels_path, config_path, sidecar]
    baseline = parent._read_frame(baseline_path)
    candidate, mask = _apply_rule(baseline)
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
            "formula": config["immutable_single_rule"],
            "baseline": describe_file(baseline_path),
            "candidate": describe_file(candidate_path),
            "mask": describe_file(mask_path),
            "scaled_rows": {group: int(mask[group].sum()) for group in TARGET_COLS},
            "2023_score_label_cells_decoded_before_lock": 0,
            "2024_or_2025_value_cells_decoded": 0,
        },
    )

    labels_2023 = parent._read_stage1_labels(labels_path).reindex(baseline.index)
    records = parent._stage1_records(labels_2023, baseline, candidate)
    component_checks = _stage1_component_checks(records)
    totals_pass = all(record["delta_total_score"] > 0.0 for record in records)
    components_pass = all(row["passed"] for row in component_checks.values())
    stage1_pass = totals_pass and components_pass
    stage1_results_path = out_dir / "stage1_results.json"
    parent._write_json(
        stage1_results_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA256,
            "candidate_lock": describe_file(stage1_lock_path),
            "comparison_count": len(records),
            "records": records,
            "full_component_checks": component_checks,
            "all_17_total_strict_positive": totals_pass,
            "all_group_full_components_nonnegative": components_pass,
            "minimum_delta": min(record["delta_total_score"] for record in records),
            "positive_count": sum(record["delta_total_score"] > 0.0 for record in records),
            "passed": stage1_pass,
            "2024_or_2025_value_cells_decoded": 0,
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
    stage2_performed = False
    dual_pass = False
    csv_created = False
    access = {
        "2023_label_prefix_rows": 8760,
        "2024_baseline_or_label_values_read": False,
        "2025_baseline_or_sample_values_read": False,
        "csv_created": False,
    }

    if not stage1_pass:
        parent._write_json(
            stage2_results_path,
            {
                "schema_version": 1,
                "performed": False,
                "reason": "Stage1 failed total or full-component gate",
                "2024_baseline_candidate_cells_computed": 0,
                "2024_label_cells_decoded": 0,
                "dual_gate_passed": False,
            },
        )
        parent._write_json(
            final_results_path,
            {
                "schema_version": 1,
                "performed": False,
                "reason": "Stage1 rejection",
                "year_2025_baseline_or_sample_value_cells_decoded": 0,
                "csv_created": False,
            },
        )
    else:
        stage2_performed = True
        stage2_candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        stage2_files: list[Path] = []
        for spec in config["stage2_2024_if_and_only_if_stage1_passes"][
            "baselines_in_fixed_order"
        ]:
            baseline_id = str(spec["id"])
            path = parent._verify(spec)
            accessed_inputs.append(path)
            base = parent._read_frame(path)
            cand, gate = _apply_rule(base)
            cand_path = out_dir / f"stage2_{baseline_id}_candidate.parquet"
            gate_path = out_dir / f"stage2_{baseline_id}_mask.parquet"
            parent._atomic_parquet(cand, cand_path)
            parent._atomic_parquet(gate, gate_path)
            stage2_files.extend([cand_path, gate_path])
            stage2_candidates[baseline_id] = (base, cand, gate)
        stage2_lock_path = out_dir / "stage2_candidates_before_label_lock.json"
        parent._write_json(
            stage2_lock_path,
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "config_sha256": CONFIG_SHA256,
                "files": [describe_file(path) for path in stage2_files],
                "scaled_rows": {
                    key: {group: int(value[2][group].sum()) for group in TARGET_COLS}
                    for key, value in stage2_candidates.items()
                },
                "2024_score_label_cells_decoded_before_lock": 0,
            },
        )
        labels = _bounded_stage2_labels(labels_path)
        variants: dict[str, Any] = {}
        dual_pass = True
        for baseline_id, (base, cand, _) in stage2_candidates.items():
            aligned = labels.reindex(base.index).loc[:, TARGET_COLS]
            group_records, mixed_records, checks = parent._stage2_records(
                aligned, base, cand, baseline_id
            )
            passed = all(checks.values())
            dual_pass = dual_pass and passed
            variants[baseline_id] = {
                "group_record_count": len(group_records),
                "mixed_record_count": len(mixed_records),
                "group_records": group_records,
                "mixed_records": mixed_records,
                "checks": checks,
                "passed": passed,
            }
        access["2024_baseline_or_label_values_read"] = True
        parent._write_json(
            stage2_results_path,
            {
                "schema_version": 1,
                "performed": True,
                "candidate_lock": describe_file(stage2_lock_path),
                "variants": variants,
                "dual_gate_passed": dual_pass,
            },
        )
        outputs.extend([*stage2_files, stage2_lock_path])
        if not dual_pass:
            parent._write_json(
                final_results_path,
                {
                    "schema_version": 1,
                    "performed": False,
                    "reason": "dual Stage2 rejection",
                    "year_2025_baseline_or_sample_value_cells_decoded": 0,
                    "csv_created": False,
                },
            )
        else:
            final_spec = config["final_if_and_only_if_dual_stage2_passes"]
            final_lock_path = out_dir / "final_prescore_lock.json"
            parent._write_json(
                final_lock_path,
                {
                    "schema_version": 1,
                    "created_utc": utc_now(),
                    "config_sha256": CONFIG_SHA256,
                    "stage1": describe_file(stage1_results_path),
                    "stage2": describe_file(stage2_results_path),
                    "baseline_identity": final_spec["baseline"],
                    "sample_identity": final_spec["sample"],
                    "year_2025_baseline_or_sample_value_cells_decoded_before_lock": 0,
                },
            )
            final_baseline_path = parent._verify(final_spec["baseline"])
            sample_path = parent._verify(final_spec["sample"])
            accessed_inputs.extend([final_baseline_path, sample_path])
            final_baseline = parent._read_frame(final_baseline_path)
            final_candidate, final_mask = _apply_rule(final_baseline)
            sample = pd.read_csv(sample_path, encoding="utf-8-sig")
            if len(sample) != len(final_candidate) or list(sample.columns[2:]) != list(TARGET_COLS):
                raise AssertionError("sample schema differs")
            submission = sample.copy()
            for group in TARGET_COLS:
                submission[group] = final_candidate[group].to_numpy()
            csv_path = out_dir / str(final_spec["csv"])
            final_candidate_path = out_dir / "final_candidate_2025.parquet"
            final_mask_path = out_dir / "final_mask_2025.parquet"
            parent._atomic_parquet(final_candidate, final_candidate_path)
            parent._atomic_parquet(final_mask, final_mask_path)
            parent._atomic_csv(submission, csv_path)
            readback = pd.read_csv(csv_path, encoding="utf-8-sig")
            if len(readback) != 8760 or list(readback.columns) != list(sample.columns):
                raise AssertionError("CSV readback schema differs")
            for group in TARGET_COLS:
                expected = np.round(final_candidate[group].to_numpy(np.float64), 6)
                np.testing.assert_array_equal(readback[group].to_numpy(np.float64), expected)
                if not np.isfinite(expected).all() or np.any(expected < 0.0) or np.any(
                    expected > 1.02 * CAPACITY_KWH[group]
                ):
                    raise AssertionError("final values out of bounds")
            if not csv_path.read_bytes().startswith(b"\xef\xbb\xbf"):
                raise AssertionError("CSV BOM absent")
            csv_created = True
            access["2025_baseline_or_sample_values_read"] = True
            access["csv_created"] = True
            outputs.extend(
                [final_lock_path, final_candidate_path, final_mask_path, csv_path]
            )
            parent._write_json(
                final_results_path,
                {
                    "schema_version": 1,
                    "performed": True,
                    "csv_created": True,
                    "scaled_rows": {
                        group: int(final_mask[group].sum()) for group in TARGET_COLS
                    },
                    "csv": describe_file(csv_path),
                    "bom_schema_rows_bounds_and_roundtrip": True,
                },
            )

    outputs.extend([stage2_results_path, final_results_path])
    manifest_path = out_dir / "manifest.json"
    parent._write_json(
        manifest_path,
        _manifest(config, out_dir, outputs, accessed_inputs, access),
    )
    return {
        "stage1_passed": stage1_pass,
        "stage1_positive_count": sum(
            record["delta_total_score"] > 0.0 for record in records
        ),
        "stage1_minimum_delta": min(record["delta_total_score"] for record in records),
        "stage2_performed": stage2_performed,
        "dual_stage2_passed": dual_pass,
        "csv_created": csv_created,
        "out_dir": str(out_dir),
        "manifest_sha256": sha256_file(manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(parent._json_ready(run(parse_args(argv))), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

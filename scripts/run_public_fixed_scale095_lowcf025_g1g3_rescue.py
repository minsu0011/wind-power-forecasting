"""Evaluate the frozen posthoc G1/G3-only low-CF 0.95 rescue."""

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
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402


CONFIG_SHA256 = "a526233408345235c4f489856db77eca77b6b4b62417b794a99c4188f3537c61"
FACTOR = np.float64(0.95)
CUTOFF_CF = np.float64(0.25)
ACTIVE_GROUPS = ("kpx_group_1", "kpx_group_3")
IDENTITY_GROUP = "kpx_group_2"
SEGMENTS = parent.SEGMENTS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/public_fixed_scale095_lowcf025_g1g3_rescue_preregister_v1.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/public_fixed_scale095_lowcf025_g1g3_rescue_v1"),
    )
    return parser.parse_args(argv)


def apply_rescue(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = baseline.copy(deep=True)
    mask = pd.DataFrame(False, index=baseline.index, columns=TARGET_COLS)
    for group in ACTIVE_GROUPS:
        base = baseline[group].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(base).all():
            raise AssertionError(f"non-finite active baseline: {group}")
        gate = (base / np.float64(CAPACITY_KWH[group])) <= CUTOFF_CF
        values = base.copy()
        values[gate] = FACTOR * base[gate]
        if not np.array_equal(values[~gate].view(np.uint64), base[~gate].view(np.uint64)):
            raise AssertionError(f"ungated identity bits differ: {group}")
        candidate[group] = values
        mask[group] = gate
    base_identity = baseline[IDENTITY_GROUP].to_numpy(dtype=np.float64, copy=False)
    cand_identity = candidate[IDENTITY_GROUP].to_numpy(dtype=np.float64, copy=False)
    if not np.array_equal(base_identity.view(np.uint64), cand_identity.view(np.uint64)):
        raise AssertionError("G2 identity bits differ")
    values = candidate.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise AssertionError("candidate contains non-finite values")
    for group in TARGET_COLS:
        if candidate[group].min() < 0.0 or candidate[group].max() > 1.02 * CAPACITY_KWH[group]:
            raise AssertionError(f"candidate bounds differ: {group}")
    return candidate, mask


def _identity_checks(
    baseline: pd.DataFrame, candidate: pd.DataFrame
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for segment in SEGMENTS:
        rows = baseline.index[parent._segment_mask(baseline.index, segment)]
        base = np.ascontiguousarray(baseline.loc[rows, IDENTITY_GROUP].to_numpy(dtype=np.float64))
        cand = np.ascontiguousarray(candidate.loc[rows, IDENTITY_GROUP].to_numpy(dtype=np.float64))
        checks.append(
            {
                "segment": segment,
                "bit_exact": bool(np.array_equal(base.view(np.uint64), cand.view(np.uint64))),
            }
        )
    return checks


def _score_variant(
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_id: str,
) -> dict[str, Any]:
    group_records: list[dict[str, Any]] = []
    mixed_records: list[dict[str, Any]] = []
    for group in ACTIVE_GROUPS:
        for segment in SEGMENTS:
            rows = baseline.index[parent._segment_mask(baseline.index, segment)]
            base = parent._group_score(labels.loc[rows, group], baseline.loc[rows, group], group)
            cand = parent._group_score(labels.loc[rows, group], candidate.loc[rows, group], group)
            group_records.append(
                {
                    "baseline_id": baseline_id,
                    "group": group,
                    "segment": segment,
                    "delta_total_score": cand["total_score"] - base["total_score"],
                    "delta_one_minus_nmae": cand["one_minus_nmae"] - base["one_minus_nmae"],
                    "delta_ficr": cand["ficr"] - base["ficr"],
                }
            )
    for segment in SEGMENTS:
        rows = baseline.index[parent._segment_mask(baseline.index, segment)]
        base = score_details(labels.loc[rows, TARGET_COLS], baseline.loc[rows, TARGET_COLS]).as_dict()
        cand = score_details(labels.loc[rows, TARGET_COLS], candidate.loc[rows, TARGET_COLS]).as_dict()
        mixed_records.append(
            {
                "baseline_id": baseline_id,
                "segment": segment,
                "delta_total_score": cand["total_score"] - base["total_score"],
                "delta_one_minus_nmae": cand["one_minus_nmae"] - base["one_minus_nmae"],
                "delta_ficr": cand["ficr"] - base["ficr"],
            }
        )
    identity = _identity_checks(baseline, candidate)
    full = next(record for record in mixed_records if record["segment"] == "full")
    checks = {
        "all_14_active_group_deltas_strict_positive": all(
            record["delta_total_score"] > 0.0 for record in group_records
        ),
        "all_7_mixed_deltas_strict_positive": all(
            record["delta_total_score"] > 0.0 for record in mixed_records
        ),
        "mixed_full_one_minus_nmae_nonnegative": full["delta_one_minus_nmae"] >= 0.0,
        "mixed_full_ficr_nonnegative": full["delta_ficr"] >= 0.0,
        "g2_identity_all_7": all(record["bit_exact"] for record in identity),
    }
    return {
        "active_group_records": group_records,
        "mixed_records": mixed_records,
        "g2_identity_checks": identity,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def _manifest(
    config: Mapping[str, Any], out_dir: Path, outputs: Sequence[Path]
) -> dict[str, Any]:
    specs: list[Mapping[str, Any]] = [
        config["duplicate_census"],
        *config["immutable_parent_lineage"].values(),
        config["stage2_2024"]["labels"],
        *config["stage2_2024"]["baselines_in_fixed_order"],
        config["final_if_and_only_if_dual_stage2_passes"]["baseline"],
        config["final_if_and_only_if_dual_stage2_passes"]["sample"],
    ]
    specs = [spec for spec in specs if isinstance(spec, Mapping) and "path" in spec]
    closure = parent._ast_closure(Path(__file__))
    test = PROJECT_DIR / "tests/test_public_fixed_scale095_lowcf025_g1g3_rescue.py"
    config_path = PROJECT_DIR / "configs/public_fixed_scale095_lowcf025_g1g3_rescue_preregister_v1.json"
    sidecar = config_path.with_suffix(".sha256")
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA256,
        "risk_classification": config["risk_classification"],
        "formula": config["immutable_single_rescue"],
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
    for spec in config["immutable_parent_lineage"].values():
        if isinstance(spec, Mapping) and "path" in spec:
            parent._verify(spec)
    label_path = parent._verify(config["stage2_2024"]["labels"])
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    _copy(config_path, out_dir / "preregister.json")
    _copy(sidecar, out_dir / "preregister.sha256")

    variants: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    candidate_files: list[Path] = []
    for spec in config["stage2_2024"]["baselines_in_fixed_order"]:
        baseline_id = str(spec["id"])
        baseline = parent._read_frame(parent._verify(spec))
        candidate, mask = apply_rescue(baseline)
        candidate_path = out_dir / f"stage2_{baseline_id}_candidate.parquet"
        mask_path = out_dir / f"stage2_{baseline_id}_mask.parquet"
        parent._atomic_parquet(candidate, candidate_path)
        parent._atomic_parquet(mask, mask_path)
        candidate_files.extend([candidate_path, mask_path])
        variants[baseline_id] = (baseline, candidate, mask)
    candidate_lock_path = out_dir / "stage2_candidates_before_label_lock.json"
    parent._write_json(
        candidate_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA256,
            "files": [describe_file(path) for path in candidate_files],
            "gate_rows": {
                baseline_id: {group: int(mask[group].sum()) for group in ACTIVE_GROUPS}
                for baseline_id, (_, _, mask) in variants.items()
            },
            "g2_identity_bits": True,
            "2024_score_label_cells_decoded_before_lock": 0,
        },
    )
    labels = pd.read_csv(label_path, encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    results: dict[str, Any] = {}
    dual_pass = True
    for baseline_id, (baseline, candidate, _) in variants.items():
        aligned = labels.reindex(baseline.index).loc[:, TARGET_COLS]
        result = _score_variant(aligned, baseline, candidate, baseline_id)
        results[baseline_id] = result
        dual_pass = dual_pass and bool(result["passed"])
    stage2_results_path = out_dir / "stage2_results.json"
    parent._write_json(
        stage2_results_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA256,
            "candidate_lock": describe_file(candidate_lock_path),
            "variants": results,
            "dual_gate_passed": dual_pass,
            "no_retune_or_second_rescue": True,
        },
    )
    outputs: list[Path] = [
        out_dir / "preregister.json",
        out_dir / "preregister.sha256",
        *candidate_files,
        candidate_lock_path,
        stage2_results_path,
    ]
    final_results_path = out_dir / "final_results.json"
    if not dual_pass:
        parent._write_json(
            final_results_path,
            {
                "schema_version": 1,
                "performed": False,
                "reason": "dual Stage2 rejection",
                "year_2025_baseline_or_sample_cells": 0,
                "csv_created": False,
            },
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
                "year_2025_baseline_or_sample_cells_before_lock": 0,
            },
        )
        baseline = parent._read_frame(parent._verify(final_cfg["baseline"]))
        candidate, mask = apply_rescue(baseline)
        sample = pd.read_csv(parent._verify(final_cfg["sample"]), encoding="utf-8-sig")
        if len(sample) != len(candidate) or list(sample.columns[1:]) != list(TARGET_COLS):
            raise AssertionError("sample schema differs")
        parsed_time = pd.to_datetime(sample.iloc[:, 0])
        if not pd.DatetimeIndex(parsed_time).equals(candidate.index):
            raise AssertionError("sample timestamp order differs")
        submission = sample.copy()
        for group in TARGET_COLS:
            submission[group] = candidate[group].to_numpy(dtype=np.float64)
        candidate_path = out_dir / "final_candidate_2025.parquet"
        mask_path = out_dir / "final_mask_2025.parquet"
        csv_path = out_dir / str(final_cfg["csv"])
        parent._atomic_parquet(candidate, candidate_path)
        parent._atomic_parquet(mask, mask_path)
        parent._atomic_csv(submission, csv_path)
        if not csv_path.read_bytes().startswith(b"\xef\xbb\xbf"):
            raise AssertionError("CSV BOM differs")
        roundtrip = pd.read_csv(csv_path, encoding="utf-8-sig")
        for group in TARGET_COLS:
            expected = np.round(candidate[group].to_numpy(dtype=np.float64), 6)
            np.testing.assert_array_equal(roundtrip[group].to_numpy(dtype=np.float64), expected)
        outputs.extend([final_lock_path, candidate_path, mask_path, csv_path])
        parent._write_json(
            final_results_path,
            {
                "schema_version": 1,
                "performed": True,
                "csv_created": True,
                "gate_rows": {group: int(mask[group].sum()) for group in ACTIVE_GROUPS},
                "g2_identity_bits": True,
                "csv": describe_file(csv_path),
            },
        )
    outputs.append(final_results_path)
    manifest_path = out_dir / "manifest.json"
    parent._write_json(manifest_path, _manifest(config, out_dir, outputs))
    return {
        "dual_gate_passed": dual_pass,
        "csv_created": json.loads(final_results_path.read_text(encoding="utf-8"))["csv_created"],
        "manifest_sha256": sha256_file(manifest_path),
        "out_dir": str(out_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(parent._json_ready(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

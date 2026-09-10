"""Run the frozen Public-adaptive 0.95 scale only at baseline CF <= 0.50."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
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
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


CONFIG_SHA256 = "5d6bbe9620860488822d6316bccabcc7697bd9196c97b9ecd76a4730c657f6f6"
FACTOR = np.float64(0.95)
CUTOFF_CF = np.float64(0.50)
BASE_CONFIG_SHA256 = "34785ae822c439b57b60059ce1979a8593fdebc84e811d6095d70aaae57be586"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/public_fixed_scale095_lowcf050_preregister_v1.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/public_fixed_scale095_lowcf050_v1"),
    )
    return parser.parse_args(argv)


def _absolute(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _verify_record(record: Mapping[str, Any]) -> Path:
    path = _absolute(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_size = record.get("bytes", record.get("size_bytes"))
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise AssertionError(f"size differs: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise AssertionError(f"hash differs: {path}")
    return path


def _apply_rule(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = baseline.copy(deep=True)
    masks = pd.DataFrame(False, index=baseline.index, columns=TARGET_COLS)
    for group in TARGET_COLS:
        base = baseline[group].to_numpy(dtype=np.float64, copy=True)
        mask = np.isfinite(base) & (
            base / np.float64(CAPACITY_KWH[group]) <= CUTOFF_CF
        )
        values = base.copy()
        values[mask] = FACTOR * base[mask]
        if not np.array_equal(
            values[~mask].view(np.uint64), base[~mask].view(np.uint64)
        ):
            raise AssertionError(f"identity bits differ: {group}")
        candidate[group] = values
        masks[group] = mask
    return candidate, masks


def _local_python_closure(entry: Path) -> tuple[Path, ...]:
    pending = [entry.resolve()]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file() or PROJECT_DIR not in path.parents:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            if module:
                candidate = PROJECT_DIR / (module.replace(".", "/") + ".py")
                package = PROJECT_DIR / module.replace(".", "/") / "__init__.py"
                if candidate.is_file():
                    pending.append(candidate.resolve())
                if package.is_file():
                    pending.append(package.resolve())
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    candidate = PROJECT_DIR / (alias.name.replace(".", "/") + ".py")
                    package = PROJECT_DIR / alias.name.replace(".", "/") / "__init__.py"
                    if candidate.is_file():
                        pending.append(candidate.resolve())
                    if package.is_file():
                        pending.append(package.resolve())
    return tuple(sorted(seen))


def _merged_parent_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    base_path = _absolute(config["scientific_lineage"]["base_contract"]["path"])
    if sha256_file(base_path) != BASE_CONFIG_SHA256:
        raise AssertionError("base contract hash differs")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    merged = copy.deepcopy(base)
    merged["experiment_id"] = config["experiment_id"]
    merged["risk_classification"] = config["risk_classification"]
    merged["immutable_single_rule"] = config["immutable_single_rule"]
    merged["duplicate_census"] = config["scientific_lineage"]["duplicate_census"]
    merged["cutoff_basis_only"] = config["scientific_lineage"]["cutoff_source"]
    merged["final_if_and_only_if_dual_stage2_passes"]["csv"] = config[
        "final_if_dual_gate_passes"
    ]["csv"]
    return merged


def _manifest(
    config: Mapping[str, Any], merged: Mapping[str, Any], out_dir: Path, outputs: list[Path]
) -> dict[str, Any]:
    explicit = [
        _absolute(config["scientific_lineage"]["base_contract"]["path"]),
        _absolute(config["scientific_lineage"]["duplicate_census"]["path"]),
        _absolute(config["scientific_lineage"]["cutoff_source"]["path"]),
        _absolute(merged["stage1_2023"]["baseline"]["path"]),
        _absolute(merged["stage1_2023"]["labels"]["path"]),
        *[
            _absolute(item["path"])
            for item in merged["stage2_2024_if_and_only_if_stage1_passes"][
                "baselines_in_fixed_order"
            ]
        ],
        _absolute(merged["final_if_and_only_if_dual_stage2_passes"]["baseline"]["path"]),
        _absolute(merged["final_if_and_only_if_dual_stage2_passes"]["sample"]["path"]),
        _absolute("configs/public_fixed_scale095_lowcf050_preregister_v1.json"),
        _absolute("configs/public_fixed_scale095_lowcf050_preregister_v1.sha256"),
        _absolute("tests/test_public_fixed_scale095_lowcf050.py"),
    ]
    closure = _local_python_closure(Path(__file__))
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA256,
        "risk_classification": config["risk_classification"],
        "formula": config["immutable_single_rule"],
        "runtime": {"python": sys.version, "packages": package_versions()},
        "git": git_state(PROJECT_DIR),
        "source_closure": [describe_file(path) for path in closure],
        "inputs": [describe_file(path) for path in explicit],
        "outputs": [describe_file(path) for path in outputs],
        "canonical_directory": str(out_dir.resolve()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    if sha256_file(config_path) != CONFIG_SHA256:
        raise AssertionError("config hash differs")
    sidecar = config_path.with_suffix(".sha256")
    expected = f"{CONFIG_SHA256}  {config_path.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected:
        raise AssertionError("config sidecar differs")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    merged = _merged_parent_contract(config)
    _verify_record(config["scientific_lineage"]["duplicate_census"])
    _verify_record(config["scientific_lineage"]["cutoff_source"])
    baseline_path = _verify_record(merged["stage1_2023"]["baseline"])
    labels_path = _verify_record(merged["stage1_2023"]["labels"])
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    parent._copy(config_path, out_dir / "preregister.json")
    parent._copy(sidecar, out_dir / "preregister.sha256")

    baseline = parent._read_frame(baseline_path)
    candidate, mask = _apply_rule(baseline)
    candidate_path = out_dir / "stage1_candidate_2023.parquet"
    mask_path = out_dir / "stage1_mask_2023.parquet"
    parent._atomic_parquet(candidate, candidate_path)
    parent._atomic_parquet(mask, mask_path)
    lock_path = out_dir / "stage1_candidate_before_label_lock.json"
    parent._write_json(lock_path, {
        "schema_version": 1,
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA256,
        "formula": config["immutable_single_rule"],
        "candidate": describe_file(candidate_path),
        "mask": describe_file(mask_path),
        "gate_rows": {g: int(mask[g].sum()) for g in TARGET_COLS},
        "2023_score_label_cells_decoded_before_lock": 0,
        "2024_prediction_or_label_cells_decoded_for_this_run": 0,
    })
    labels_2023 = parent._read_stage1_labels(labels_path).reindex(baseline.index)
    records = parent._stage1_records(labels_2023, baseline, candidate)
    stage1_pass = all(row["delta_total_score"] > 0.0 for row in records)
    stage1_path = out_dir / "stage1_results.json"
    parent._write_json(stage1_path, {
        "schema_version": 1,
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA256,
        "candidate_lock": describe_file(lock_path),
        "comparison_count": len(records),
        "records": records,
        "minimum_delta": min(row["delta_total_score"] for row in records),
        "positive_count": sum(row["delta_total_score"] > 0.0 for row in records),
        "passed": stage1_pass,
        "2024_prediction_or_label_cells_decoded_for_this_run": 0,
    })
    outputs = [out_dir / "preregister.json", out_dir / "preregister.sha256", candidate_path, mask_path, lock_path, stage1_path]
    stage2_path = out_dir / "stage2_results.json"
    final_path = out_dir / "final_results.json"
    if not stage1_pass:
        parent._write_json(stage2_path, {"schema_version": 1, "performed": False, "reason": "Stage1 strict rejection", "2024_baseline_candidate_cells_computed": 0, "2024_label_cells_decoded_for_this_run": 0, "dual_gate_passed": False})
        parent._write_json(final_path, {"schema_version": 1, "performed": False, "reason": "Stage1 rejection", "year_2025_baseline_or_sample_value_cells_decoded_for_this_run": 0, "csv_created": False})
    else:
        stage2_candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        stage2_files: list[Path] = []
        for spec in merged["stage2_2024_if_and_only_if_stage1_passes"]["baselines_in_fixed_order"]:
            name = str(spec["id"])
            base = parent._read_frame(_verify_record(spec))
            cand, gate = _apply_rule(base)
            cand_path = out_dir / f"stage2_{name}_candidate.parquet"
            gate_path = out_dir / f"stage2_{name}_mask.parquet"
            parent._atomic_parquet(cand, cand_path)
            parent._atomic_parquet(gate, gate_path)
            stage2_files += [cand_path, gate_path]
            stage2_candidates[name] = (base, cand, gate)
        stage2_lock = out_dir / "stage2_candidates_before_label_lock.json"
        parent._write_json(stage2_lock, {"schema_version": 1, "created_utc": utc_now(), "config_sha256": CONFIG_SHA256, "files": [describe_file(path) for path in stage2_files], "2024_score_label_cells_decoded_before_lock": 0})
        labels = pd.read_csv(labels_path, encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
        variants: dict[str, Any] = {}
        dual_pass = True
        for name, (base, cand, _) in stage2_candidates.items():
            aligned = labels.reindex(base.index).loc[:, TARGET_COLS]
            group_rows, mixed_rows, checks = parent._stage2_records(aligned, base, cand, name)
            passed = all(checks.values())
            dual_pass &= passed
            variants[name] = {"group_records": group_rows, "mixed_records": mixed_rows, "checks": checks, "passed": passed}
        parent._write_json(stage2_path, {"schema_version": 1, "performed": True, "candidate_lock": describe_file(stage2_lock), "variants": variants, "dual_gate_passed": dual_pass})
        outputs += stage2_files + [stage2_lock]
        if not dual_pass:
            parent._write_json(final_path, {"schema_version": 1, "performed": False, "reason": "dual Stage2 rejection", "year_2025_baseline_or_sample_value_cells_decoded_for_this_run": 0, "csv_created": False})
        else:
            final_spec = merged["final_if_and_only_if_dual_stage2_passes"]
            final_lock = out_dir / "final_prescore_lock.json"
            parent._write_json(final_lock, {"schema_version": 1, "created_utc": utc_now(), "config_sha256": CONFIG_SHA256, "stage1": describe_file(stage1_path), "stage2": describe_file(stage2_path), "year_2025_baseline_or_sample_value_cells_decoded_before_lock": 0})
            final_baseline = parent._read_frame(_verify_record(final_spec["baseline"]))
            final_candidate, final_mask = _apply_rule(final_baseline)
            sample = pd.read_csv(_verify_record(final_spec["sample"]), encoding="utf-8-sig")
            if len(sample) != len(final_candidate) or list(sample.columns[1:]) != list(TARGET_COLS):
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
            outputs += [final_lock, final_candidate_path, final_mask_path, csv_path]
            parent._write_json(final_path, {"schema_version": 1, "performed": True, "csv_created": True, "gate_rows": {g: int(final_mask[g].sum()) for g in TARGET_COLS}, "csv": describe_file(csv_path)})
    outputs += [stage2_path, final_path]
    manifest_path = out_dir / "manifest.json"
    parent._write_json(manifest_path, _manifest(config, merged, out_dir, outputs))
    return {"stage1_passed": stage1_pass, "stage1_positive_count": sum(row["delta_total_score"] > 0.0 for row in records), "stage1_minimum_delta": min(row["delta_total_score"] for row in records), "manifest_sha256": sha256_file(manifest_path), "out_dir": str(out_dir)}


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

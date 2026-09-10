"""Evaluate the frozen Public-adaptive 0.95 scale below baseline CF 0.25."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.metric import (  # noqa: E402
    CAPACITY_KWH,
    TARGET_COLS,
    group_metrics,
    score_details,
)


CONFIG_SHA256 = "34785ae822c439b57b60059ce1979a8593fdebc84e811d6095d70aaae57be586"
FACTOR = np.float64(0.95)
CUTOFF_CF = np.float64(0.25)
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/public_fixed_scale095_lowcf025_preregister_v1.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/public_fixed_scale095_lowcf025_v1"),
    )
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(path, _json_ready(payload), overwrite=False)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, index=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
            lineterminator="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def _absolute(spec: Mapping[str, Any]) -> Path:
    path = Path(str(spec["path"]))
    return path if path.is_absolute() else PROJECT_DIR / path


def _verify(spec: Mapping[str, Any]) -> Path:
    path = _absolute(spec)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_size = spec.get("bytes", spec.get("size_bytes"))
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise AssertionError(f"size differs: {path}")
    if sha256_file(path) != str(spec["sha256"]):
        raise AssertionError(f"hash differs: {path}")
    return path


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        time_column = "forecast_kst_dtm" if "forecast_kst_dtm" in frame else "kst_dtm"
        frame = frame.set_index(pd.to_datetime(frame.pop(time_column)))
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise AssertionError(f"invalid time index: {path}")
    if tuple(frame.columns) != tuple(TARGET_COLS):
        raise AssertionError(f"target schema differs: {path}")
    return frame.astype(np.float64)


def _apply_rule(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = baseline.copy(deep=True)
    masks = pd.DataFrame(False, index=baseline.index, columns=TARGET_COLS)
    for group in TARGET_COLS:
        base = baseline[group].to_numpy(dtype=np.float64, copy=True)
        valid = np.isfinite(base)
        mask = valid & ((base / np.float64(CAPACITY_KWH[group])) <= CUTOFF_CF)
        values = base.copy()
        values[mask] = FACTOR * base[mask]
        identity = ~mask
        if not np.array_equal(values[identity].view(np.uint64), base[identity].view(np.uint64)):
            raise AssertionError(f"identity bits differ: {group}")
        candidate[group] = values
        masks[group] = mask
    return candidate, masks


def _segment_mask(index: pd.DatetimeIndex, segment: str) -> np.ndarray:
    if segment in {"full", "full_H2"}:
        return np.ones(len(index), dtype=bool)
    year = int(index[0].year)
    ranges = {
        "H1": (f"{year}-01-01 01:00", f"{year}-07-01 00:00"),
        "H2": (f"{year}-07-01 01:00", f"{year + 1}-01-01 00:00"),
        "Q1": (f"{year}-01-01 01:00", f"{year}-04-01 00:00"),
        "Q2": (f"{year}-04-01 01:00", f"{year}-07-01 00:00"),
        "Q3": (f"{year}-07-01 01:00", f"{year}-10-01 00:00"),
        "Q4": (f"{year}-10-01 01:00", f"{year + 1}-01-01 00:00"),
    }
    if segment in ranges:
        start, end = ranges[segment]
        return np.asarray(index.isin(pd.date_range(start, end, freq="h")))
    raise ValueError(segment)


def _group_score(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    record = group_metrics(
        actual,
        prediction,
        CAPACITY_KWH[group],
        group_name=group,
    ).as_dict()
    record["total_score"] = 0.5 * (record["one_minus_nmae"] + record["ficr"])
    return record


def _stage1_records(
    labels: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    slices = {
        "kpx_group_1": SEGMENTS,
        "kpx_group_2": SEGMENTS,
        "kpx_group_3": ("full_H2", "Q3", "Q4"),
    }
    for group in TARGET_COLS:
        available = baseline[group].notna() & labels[group].notna()
        index = baseline.index[available]
        for segment in slices[group]:
            rows = index[_segment_mask(index, segment)]
            base = _group_score(labels.loc[rows, group], baseline.loc[rows, group], group)
            cand = _group_score(labels.loc[rows, group], candidate.loc[rows, group], group)
            records.append(
                {
                    "group": group,
                    "segment": segment,
                    "rows": len(rows),
                    "baseline": base,
                    "candidate": cand,
                    "delta_total_score": cand["total_score"] - base["total_score"],
                    "delta_one_minus_nmae": cand["one_minus_nmae"] - base["one_minus_nmae"],
                    "delta_ficr": cand["ficr"] - base["ficr"],
                }
            )
    if len(records) != 17:
        raise AssertionError("Stage1 comparison count differs")
    return records


def _stage2_records(
    labels: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame, baseline_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    group_records: list[dict[str, Any]] = []
    mixed_records: list[dict[str, Any]] = []
    for group in TARGET_COLS:
        for segment in SEGMENTS:
            rows = baseline.index[_segment_mask(baseline.index, segment)]
            base = _group_score(labels.loc[rows, group], baseline.loc[rows, group], group)
            cand = _group_score(labels.loc[rows, group], candidate.loc[rows, group], group)
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
        rows = baseline.index[_segment_mask(baseline.index, segment)]
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
    full = next(record for record in mixed_records if record["segment"] == "full")
    checks = {
        "all_21_group_total_strict_positive": all(
            record["delta_total_score"] > 0.0 for record in group_records
        ),
        "all_7_mixed_total_strict_positive": all(
            record["delta_total_score"] > 0.0 for record in mixed_records
        ),
        "mixed_full_one_minus_nmae_nonnegative": full["delta_one_minus_nmae"] >= 0.0,
        "mixed_full_ficr_nonnegative": full["delta_ficr"] >= 0.0,
    }
    return group_records, mixed_records, checks


def _read_stage1_labels(path: Path) -> pd.DataFrame:
    # Exact 2022+2023 physical prefix: 8760+8760 rows. No 2024 row is parsed.
    labels = pd.read_csv(
        path,
        encoding="utf-8-sig",
        nrows=17_520,
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    return labels.iloc[-8_760:].loc[:, TARGET_COLS]


def _module_files(module: str) -> set[Path]:
    parts = [part for part in module.split(".") if part]
    if not parts:
        return set()
    found: set[Path] = set()
    file = PROJECT_DIR.joinpath(*parts).with_suffix(".py")
    package = PROJECT_DIR.joinpath(*parts, "__init__.py")
    if file.is_file():
        found.add(file.resolve())
    if package.is_file():
        found.add(package.resolve())
    for depth in range(1, len(parts)):
        init = PROJECT_DIR.joinpath(*parts[:depth], "__init__.py")
        if init.is_file():
            found.add(init.resolve())
    return found


def _ast_closure(entrypoint: Path) -> tuple[Path, ...]:
    pending = [entrypoint.resolve()]
    seen: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        tree = ast.parse(current.read_text(encoding="utf-8"), filename=str(current))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pending.extend(_module_files(alias.name))
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                pending.extend(_module_files(base))
                for alias in node.names:
                    if alias.name != "*":
                        pending.extend(_module_files(f"{base}.{alias.name}"))
    return tuple(sorted(seen))


def _manifest(
    config: Mapping[str, Any], out_dir: Path, outputs: Sequence[Path]
) -> dict[str, Any]:
    closure = _ast_closure(Path(__file__))
    test_path = PROJECT_DIR / "tests/test_public_fixed_scale095_lowcf025.py"
    inputs = [
        _absolute(config["duplicate_census"]),
        _absolute(config["cutoff_basis_only"]),
        _absolute(config["stage1_2023"]["baseline"]),
        _absolute(config["stage1_2023"]["labels"]),
        *[
            _absolute(spec)
            for spec in config["stage2_2024_if_and_only_if_stage1_passes"]["baselines_in_fixed_order"]
        ],
        _absolute(config["final_if_and_only_if_dual_stage2_passes"]["baseline"]),
        _absolute(config["final_if_and_only_if_dual_stage2_passes"]["sample"]),
        PROJECT_DIR / "configs/public_fixed_scale095_lowcf025_preregister_v1.json",
        PROJECT_DIR / "configs/public_fixed_scale095_lowcf025_preregister_v1.sha256",
    ]
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
            "test": describe_file(test_path),
        },
        "inputs": [describe_file(path) for path in inputs],
        "outputs": [describe_file(path) for path in outputs],
        "canonical_directory": str(out_dir.resolve()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    if sha256_file(config_path) != CONFIG_SHA256:
        raise AssertionError("config hash differs")
    sidecar = config_path.with_suffix(".sha256")
    expected_sidecar = f"{CONFIG_SHA256}  {config_path.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise AssertionError("config sidecar differs")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify(config["duplicate_census"])
    _verify(config["cutoff_basis_only"])
    baseline_path = _verify(config["stage1_2023"]["baseline"])
    labels_path = _verify(config["stage1_2023"]["labels"])
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    _copy(config_path, out_dir / "preregister.json")
    _copy(sidecar, out_dir / "preregister.sha256")

    baseline = _read_frame(baseline_path)
    candidate, mask = _apply_rule(baseline)
    candidate_path = out_dir / "stage1_candidate_2023.parquet"
    mask_path = out_dir / "stage1_mask_2023.parquet"
    _atomic_parquet(candidate, candidate_path)
    _atomic_parquet(mask, mask_path)
    stage1_lock_path = out_dir / "stage1_candidate_before_label_lock.json"
    _write_json(
        stage1_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA256,
            "formula": config["immutable_single_rule"],
            "baseline": describe_file(baseline_path),
            "candidate": describe_file(candidate_path),
            "mask": describe_file(mask_path),
            "gate_rows": {group: int(mask[group].sum()) for group in TARGET_COLS},
            "2023_score_label_cells_decoded_before_lock": 0,
            "2024_prediction_or_label_cells_decoded_for_this_run": 0,
        },
    )

    labels_2023 = _read_stage1_labels(labels_path).reindex(baseline.index)
    records = _stage1_records(labels_2023, baseline, candidate)
    stage1_pass = all(record["delta_total_score"] > 0.0 for record in records)
    stage1_results_path = out_dir / "stage1_results.json"
    _write_json(
        stage1_results_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA256,
            "candidate_lock": describe_file(stage1_lock_path),
            "comparison_count": len(records),
            "records": records,
            "minimum_delta": min(record["delta_total_score"] for record in records),
            "positive_count": sum(record["delta_total_score"] > 0.0 for record in records),
            "passed": stage1_pass,
            "2024_prediction_or_label_cells_decoded_for_this_run": 0,
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
        _write_json(
            stage2_results_path,
            {
                "schema_version": 1,
                "performed": False,
                "reason": "Stage1 failed at least one of 17 strict comparisons",
                "2024_baseline_candidate_cells_computed": 0,
                "2024_label_cells_decoded_for_this_run": 0,
                "dual_gate_passed": False,
            },
        )
        _write_json(
            final_results_path,
            {
                "schema_version": 1,
                "performed": False,
                "reason": "Stage1 rejection",
                "year_2025_baseline_or_sample_value_cells_decoded_for_this_run": 0,
                "csv_created": False,
            },
        )
    else:
        stage2_candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        stage2_files: list[Path] = []
        for spec in config["stage2_2024_if_and_only_if_stage1_passes"]["baselines_in_fixed_order"]:
            baseline_id = str(spec["id"])
            base = _read_frame(_verify(spec))
            cand, gate = _apply_rule(base)
            cand_path = out_dir / f"stage2_{baseline_id}_candidate.parquet"
            gate_path = out_dir / f"stage2_{baseline_id}_mask.parquet"
            _atomic_parquet(cand, cand_path)
            _atomic_parquet(gate, gate_path)
            stage2_files.extend([cand_path, gate_path])
            stage2_candidates[baseline_id] = (base, cand, gate)
        stage2_lock_path = out_dir / "stage2_candidates_before_label_lock.json"
        _write_json(
            stage2_lock_path,
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "config_sha256": CONFIG_SHA256,
                "files": [describe_file(path) for path in stage2_files],
                "gate_rows": {
                    baseline_id: {group: int(gate[group].sum()) for group in TARGET_COLS}
                    for baseline_id, (_, _, gate) in stage2_candidates.items()
                },
                "2024_score_label_cells_decoded_before_lock": 0,
            },
        )
        labels = pd.read_csv(labels_path, encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
        variants: dict[str, Any] = {}
        dual_pass = True
        for baseline_id, (base, cand, _) in stage2_candidates.items():
            aligned = labels.reindex(base.index).loc[:, TARGET_COLS]
            group_records, mixed_records, checks = _stage2_records(aligned, base, cand, baseline_id)
            passed = all(checks.values())
            dual_pass = dual_pass and passed
            variants[baseline_id] = {
                "group_records": group_records,
                "mixed_records": mixed_records,
                "checks": checks,
                "passed": passed,
            }
        _write_json(
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
            _write_json(
                final_results_path,
                {
                    "schema_version": 1,
                    "performed": False,
                    "reason": "dual Stage2 rejection",
                    "year_2025_baseline_or_sample_value_cells_decoded_for_this_run": 0,
                    "csv_created": False,
                },
            )
        else:
            final_spec = config["final_if_and_only_if_dual_stage2_passes"]
            final_lock_path = out_dir / "final_prescore_lock.json"
            _write_json(
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
            final_baseline = _read_frame(_verify(final_spec["baseline"]))
            final_candidate, final_mask = _apply_rule(final_baseline)
            sample = pd.read_csv(_verify(final_spec["sample"]), encoding="utf-8-sig")
            if len(sample) != len(final_candidate) or list(sample.columns[1:]) != list(TARGET_COLS):
                raise AssertionError("sample schema differs")
            submission = sample.copy()
            for group in TARGET_COLS:
                submission[group] = final_candidate[group].to_numpy()
            csv_path = out_dir / str(final_spec["csv"])
            final_parquet_path = out_dir / "final_candidate_2025.parquet"
            final_mask_path = out_dir / "final_mask_2025.parquet"
            _atomic_parquet(final_candidate, final_parquet_path)
            _atomic_parquet(final_mask, final_mask_path)
            _atomic_csv(submission, csv_path)
            outputs.extend([final_lock_path, final_parquet_path, final_mask_path, csv_path])
            _write_json(
                final_results_path,
                {
                    "schema_version": 1,
                    "performed": True,
                    "csv_created": True,
                    "gate_rows": {group: int(final_mask[group].sum()) for group in TARGET_COLS},
                    "csv": describe_file(csv_path),
                },
            )
    outputs.extend([stage2_results_path, final_results_path])
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, _manifest(config, out_dir, outputs))
    return {
        "stage1_passed": stage1_pass,
        "stage1_positive_count": sum(record["delta_total_score"] > 0.0 for record in records),
        "stage1_minimum_delta": min(record["delta_total_score"] for record in records),
        "out_dir": str(out_dir),
        "manifest_sha256": sha256_file(manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

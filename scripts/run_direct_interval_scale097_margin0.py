"""Evaluate the frozen direct-interval .97 scale with a strict margin-0 gate."""

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

from src.direct_interval_selective_scale import interpolate_row_utility  # noqa: E402
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details  # noqa: E402


CONFIG_SHA256 = "ec0399b6a90b336a5c4af8fa2e42a1aca0f9af7a958e1e87424551378abddcf3"
FACTOR = np.float64(0.97)
MARGIN = np.float64(0.0)
ACTIVE_GROUPS = ("kpx_group_1", "kpx_group_2")
IDENTITY_GROUP = "kpx_group_3"
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/direct_interval_scale097_margin0_preregister_v1.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/direct_interval_scale097_margin0_v1"),
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
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise AssertionError(f"invalid index: {path}")
    if tuple(frame.columns) != tuple(TARGET_COLS):
        raise AssertionError(f"target schema differs: {path}")
    return frame.astype(np.float64)


def _load_utility(path: Path, expected_rows: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"p6_raw", "p8_raw", "p6", "p8", "eae", "utility", "repair_count"}:
            raise AssertionError(f"surface keys differ: {path}")
        utility = np.asarray(payload["utility"], dtype=np.float64)
    if utility.shape != (expected_rows, 103) or not np.isfinite(utility).all():
        raise AssertionError(f"utility shape/finite differs: {path}")
    return utility


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
        if np.min(values) < 0.0 or np.max(values) > 1.02 * capacity:
            raise AssertionError(f"candidate bounds differ: {group}")
        if not np.array_equal(values[~gate].view(np.uint64), base[~gate].view(np.uint64)):
            raise AssertionError(f"ungated identity bits differ: {group}")
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
    base_g3 = baseline[IDENTITY_GROUP].to_numpy(dtype=np.float64, copy=False)
    cand_g3 = candidate[IDENTITY_GROUP].to_numpy(dtype=np.float64, copy=False)
    if not np.array_equal(base_g3.view(np.uint64), cand_g3.view(np.uint64)):
        raise AssertionError("G3 identity bits differ")
    return candidate, diagnostics


def _segments(year: int) -> dict[str, pd.DatetimeIndex]:
    make = lambda start, end: pd.date_range(start, end, freq="h", name="forecast_kst_dtm")
    return {
        "full": make(f"{year}-01-01 01:00", f"{year + 1}-01-01 00:00"),
        "H1": make(f"{year}-01-01 01:00", f"{year}-07-01 00:00"),
        "H2": make(f"{year}-07-01 01:00", f"{year + 1}-01-01 00:00"),
        "Q1": make(f"{year}-01-01 01:00", f"{year}-04-01 00:00"),
        "Q2": make(f"{year}-04-01 01:00", f"{year}-07-01 00:00"),
        "Q3": make(f"{year}-07-01 01:00", f"{year}-10-01 00:00"),
        "Q4": make(f"{year}-10-01 01:00", f"{year + 1}-01-01 00:00"),
    }


def _group_score(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    result = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group).as_dict()
    result["total_score"] = 0.5 * (result["one_minus_nmae"] + result["ficr"])
    return result


def _comparison(
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    group: str,
    segment: str,
    index: pd.DatetimeIndex,
) -> dict[str, Any]:
    base = _group_score(labels.loc[index, group], baseline.loc[index, group], group)
    cand = _group_score(labels.loc[index, group], candidate.loc[index, group], group)
    return {
        "group": group,
        "segment": segment,
        "baseline": base,
        "candidate": cand,
        "delta_total_score": cand["total_score"] - base["total_score"],
        "delta_one_minus_nmae": cand["one_minus_nmae"] - base["one_minus_nmae"],
        "delta_ficr": cand["ficr"] - base["ficr"],
    }


def _active_records(
    year: int, labels: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame
) -> list[dict[str, Any]]:
    segments = _segments(year)
    return [
        _comparison(labels, baseline, candidate, group, segment, segments[segment])
        for group in ACTIVE_GROUPS
        for segment in SEGMENTS
    ]


def _g3_identity_checks(
    baseline: pd.DataFrame, candidate: pd.DataFrame, year: int, stage1: bool
) -> list[dict[str, Any]]:
    segments = _segments(year)
    names = ("H2", "Q3", "Q4") if stage1 else SEGMENTS
    checks: list[dict[str, Any]] = []
    for name in names:
        rows = segments[name]
        base = np.ascontiguousarray(baseline.loc[rows, IDENTITY_GROUP].to_numpy(dtype=np.float64))
        cand = np.ascontiguousarray(candidate.loc[rows, IDENTITY_GROUP].to_numpy(dtype=np.float64))
        checks.append(
            {
                "segment": "full_H2" if stage1 and name == "H2" else name,
                "bit_exact": bool(np.array_equal(base.view(np.uint64), cand.view(np.uint64))),
            }
        )
    return checks


def _mixed_records(
    year: int, labels: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for segment, rows in _segments(year).items():
        base = score_details(labels.loc[rows, TARGET_COLS], baseline.loc[rows, TARGET_COLS]).as_dict()
        cand = score_details(labels.loc[rows, TARGET_COLS], candidate.loc[rows, TARGET_COLS]).as_dict()
        records.append(
            {
                "segment": segment,
                "baseline": base,
                "candidate": cand,
                "delta_total_score": cand["total_score"] - base["total_score"],
                "delta_one_minus_nmae": cand["one_minus_nmae"] - base["one_minus_nmae"],
                "delta_ficr": cand["ficr"] - base["ficr"],
            }
        )
    return records


def _read_stage1_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(
        path,
        encoding="utf-8-sig",
        nrows=17_520,
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    return labels.iloc[-8_760:].loc[:, TARGET_COLS]


def _module_files(module: str) -> set[Path]:
    parts = [part for part in module.split(".") if part]
    if not parts:
        return set()
    result: set[Path] = set()
    for path in (
        PROJECT_DIR.joinpath(*parts).with_suffix(".py"),
        PROJECT_DIR.joinpath(*parts, "__init__.py"),
    ):
        if path.is_file():
            result.add(path.resolve())
    for depth in range(1, len(parts)):
        init = PROJECT_DIR.joinpath(*parts[:depth], "__init__.py")
        if init.is_file():
            result.add(init.resolve())
    return result


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
    specs: list[Mapping[str, Any]] = [
        config["duplicate_census"],
        config["factor_lineage_only"],
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
    closure = _ast_closure(Path(__file__))
    test = PROJECT_DIR / "tests/test_direct_interval_scale097_margin0.py"
    config_path = PROJECT_DIR / "configs/direct_interval_scale097_margin0_preregister_v1.json"
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
        "inputs": [describe_file(_absolute(spec)) for spec in specs]
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
    _verify(config["duplicate_census"])
    _verify(config["factor_lineage_only"])
    baseline_path = _verify(config["stage1_2023"]["baseline"])
    label_path = _verify(config["stage1_2023"]["labels"])
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    _copy(config_path, out_dir / "preregister.json")
    _copy(sidecar, out_dir / "preregister.sha256")

    baseline = _read_frame(baseline_path)
    stage1_utilities = {
        group: _load_utility(_verify(config["stage1_2023"]["surfaces"][group]), len(baseline))
        for group in ACTIVE_GROUPS
    }
    candidate, diagnostics = apply_rule(baseline, stage1_utilities)
    candidate_path = out_dir / "stage1_candidate_2023.parquet"
    diagnostic_paths = {
        group: out_dir / f"stage1_{group}_diagnostics.parquet" for group in ACTIVE_GROUPS
    }
    _atomic_parquet(candidate, candidate_path)
    for group in ACTIVE_GROUPS:
        _atomic_parquet(diagnostics[group], diagnostic_paths[group])
    stage1_lock_path = out_dir / "stage1_candidate_before_label_lock.json"
    _write_json(
        stage1_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA256,
            "candidate": describe_file(candidate_path),
            "diagnostics": {group: describe_file(path) for group, path in diagnostic_paths.items()},
            "surface_inputs": {
                group: describe_file(_absolute(spec))
                for group, spec in config["stage1_2023"]["surfaces"].items()
            },
            "gate_rows": {group: int(diagnostics[group]["gate"].sum()) for group in ACTIVE_GROUPS},
            "g3_identity_bits": True,
            "2023_score_label_cells_decoded_before_lock": 0,
            "2024_candidate_or_label_cells": 0,
        },
    )
    labels_2023 = _read_stage1_labels(label_path).reindex(baseline.index)
    active_records = _active_records(2023, labels_2023, baseline, candidate)
    identity_checks = _g3_identity_checks(baseline, candidate, 2023, True)
    stage1_pass = all(record["delta_total_score"] > 0.0 for record in active_records) and all(
        check["bit_exact"] for check in identity_checks
    )
    stage1_results_path = out_dir / "stage1_results.json"
    _write_json(
        stage1_results_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA256,
            "candidate_lock": describe_file(stage1_lock_path),
            "active_score_comparisons": active_records,
            "g3_identity_checks": identity_checks,
            "registered_check_count": len(active_records) + len(identity_checks),
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
        _write_json(
            stage2_results_path,
            {
                "schema_version": 1,
                "performed": False,
                "reason": "Stage1 failed a registered active-group or identity check",
                "year_2024_candidate_cells": 0,
                "year_2024_label_cells": 0,
                "dual_gate_passed": False,
            },
        )
        _write_json(
            final_results_path,
            {
                "schema_version": 1,
                "performed": False,
                "reason": "Stage1 rejection",
                "heavy_refit_calls": 0,
                "year_2025_or_sample_cells": 0,
                "csv_created": False,
            },
        )
    else:
        stage2_cfg = config["stage2_2024_if_and_only_if_stage1_passes"]
        _verify(stage2_cfg["surface_provenance"])
        _verify(stage2_cfg["independent_audit"])
        stage2_utilities = {
            group: _load_utility(_verify(stage2_cfg["surfaces"][group]), 8784)
            for group in ACTIVE_GROUPS
        }
        variants: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]] = {}
        stage2_files: list[Path] = []
        for spec in stage2_cfg["baselines_in_fixed_order"]:
            baseline_id = str(spec["id"])
            base = _read_frame(_verify(spec))
            cand, diag = apply_rule(base, stage2_utilities)
            cand_path = out_dir / f"stage2_{baseline_id}_candidate.parquet"
            _atomic_parquet(cand, cand_path)
            stage2_files.append(cand_path)
            for group in ACTIVE_GROUPS:
                path = out_dir / f"stage2_{baseline_id}_{group}_diagnostics.parquet"
                _atomic_parquet(diag[group], path)
                stage2_files.append(path)
            variants[baseline_id] = (base, cand, diag)
        stage2_lock_path = out_dir / "stage2_candidates_before_label_lock.json"
        _write_json(
            stage2_lock_path,
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "config_sha256": CONFIG_SHA256,
                "files": [describe_file(path) for path in stage2_files],
                "gate_rows": {
                    baseline_id: {
                        group: int(diag[group]["gate"].sum()) for group in ACTIVE_GROUPS
                    }
                    for baseline_id, (_, _, diag) in variants.items()
                },
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
            groups = _active_records(2024, aligned, base, cand)
            mixed = _mixed_records(2024, aligned, base, cand)
            identity = _g3_identity_checks(base, cand, 2024, False)
            full = next(record for record in mixed if record["segment"] == "full")
            checks = {
                "all_14_active_group_deltas_strict_positive": all(
                    record["delta_total_score"] > 0.0 for record in groups
                ),
                "all_7_mixed_deltas_strict_positive": all(
                    record["delta_total_score"] > 0.0 for record in mixed
                ),
                "mixed_full_one_minus_nmae_nonnegative": full["delta_one_minus_nmae"] >= 0.0,
                "mixed_full_ficr_nonnegative": full["delta_ficr"] >= 0.0,
                "g3_identity_all_7": all(check["bit_exact"] for check in identity),
            }
            passed = all(checks.values())
            dual_pass = dual_pass and passed
            variant_results[baseline_id] = {
                "active_group_records": groups,
                "mixed_records": mixed,
                "g3_identity_checks": identity,
                "checks": checks,
                "passed": passed,
            }
        _write_json(
            stage2_results_path,
            {
                "schema_version": 1,
                "performed": True,
                "candidate_lock": describe_file(stage2_lock_path),
                "variants": variant_results,
                "dual_gate_passed": dual_pass,
            },
        )
        outputs.extend([*stage2_files, stage2_lock_path])
        if dual_pass:
            _write_json(
                final_results_path,
                {
                    "schema_version": 1,
                    "performed": False,
                    "reason": "dual Stage2 PASS; separate guarded heavy final refit required",
                    "heavy_refit_calls": 0,
                    "year_2025_or_sample_cells": 0,
                    "csv_created": False,
                    "promotion_pending_final": True,
                },
            )
        else:
            _write_json(
                final_results_path,
                {
                    "schema_version": 1,
                    "performed": False,
                    "reason": "dual Stage2 rejection",
                    "heavy_refit_calls": 0,
                    "year_2025_or_sample_cells": 0,
                    "csv_created": False,
                },
            )
    outputs.extend([stage2_results_path, final_results_path])
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, _manifest(config, out_dir, outputs))
    stage2_payload = json.loads(stage2_results_path.read_text(encoding="utf-8"))
    return {
        "stage1_passed": stage1_pass,
        "stage1_positive_active_count": sum(
            record["delta_total_score"] > 0.0 for record in active_records
        ),
        "stage1_minimum_active_delta": min(
            record["delta_total_score"] for record in active_records
        ),
        "stage2_performed": bool(stage2_payload["performed"]),
        "dual_gate_passed": bool(stage2_payload["dual_gate_passed"]),
        "manifest_sha256": sha256_file(manifest_path),
        "out_dir": str(out_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

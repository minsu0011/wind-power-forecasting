"""Evaluate the frozen scale097 A plus fixed neural 2024 delta interaction."""

from __future__ import annotations

import argparse
import ast
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

from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now, write_json_atomic  # noqa: E402


CONFIG_SHA = "646bf9c876939a74b6f74ae9ac97cfc3c9d4575d4b650fb5559a2bcd2e2e5543"
TARGETS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
ACTIVE = TARGETS[:2]
IDENTITY = TARGETS[2]
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
EXPECTED_DELTA_ROWS = {"kpx_group_1": 397, "kpx_group_2": 243}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/public_adaptive_scale097_neural_fixed_delta_diagnostic_preregister_v1.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/postgate/public_adaptive_scale097_neural_fixed_delta_diagnostic_v1"))
    return parser.parse_args(argv)


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(path, _ready(payload), overwrite=False)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_parquet(temporary, index=True)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _absolute(spec: Mapping[str, Any]) -> Path:
    path = Path(str(spec["path"]))
    return path if path.is_absolute() else PROJECT_DIR / path


def _verify(spec: Mapping[str, Any]) -> Path:
    path = _absolute(spec)
    if not path.is_file():
        raise FileNotFoundError(path)
    size = spec.get("bytes", spec.get("size_bytes"))
    if size is not None and path.stat().st_size != int(size):
        raise AssertionError(f"size differs: {path}")
    if sha256_file(path) != str(spec["sha256"]):
        raise AssertionError(f"hash differs: {path}")
    return path


def _verify_config(path: Path) -> dict[str, Any]:
    if sha256_file(path) != CONFIG_SHA:
        raise AssertionError("config hash differs")
    if path.with_suffix(".sha256").read_text(encoding="utf-8") != f"{CONFIG_SHA}  {path.name}\n":
        raise AssertionError("sidecar differs")
    config = json.loads(path.read_text(encoding="utf-8"))
    rule = config["immutable_single_interaction"]
    if rule["scale_inherited_and_unchanged"] != 0.97 or rule["neural_weight_inherited_and_unchanged"] != 0.05 or rule["neural_margin_inherited_and_unchanged"] != 0.01:
        raise AssertionError("fixed rule differs")
    for section in ("lineage", "input_identities"):
        for spec in config[section].values():
            _verify(spec)
    return config


def _local(module: str) -> set[Path]:
    parts = [part for part in module.split(".") if part]
    if not parts:
        return set()
    result: set[Path] = set()
    file = PROJECT_DIR.joinpath(*parts).with_suffix(".py")
    package = PROJECT_DIR.joinpath(*parts, "__init__.py")
    if file.is_file():
        result.add(file.resolve())
    if package.is_file():
        result.add(package.resolve())
    for depth in range(1, len(parts)):
        init = PROJECT_DIR.joinpath(*parts[:depth], "__init__.py")
        if init.is_file():
            result.add(init.resolve())
    return result


def resolve_ast_closure(entry: Path) -> tuple[Path, ...]:
    pending = [entry.resolve()]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if PROJECT_DIR not in path.parents:
            raise AssertionError("closure escapes project")
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pending.extend(_local(alias.name))
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                pending.extend(_local(base))
                for alias in node.names:
                    if alias.name != "*":
                        pending.extend(_local(f"{base}.{alias.name}"))
    return tuple(sorted(seen))


def _closure(args: argparse.Namespace) -> dict[str, Any]:
    files = resolve_ast_closure(Path(__file__))
    return {"resolved_relative_paths": [path.relative_to(PROJECT_DIR).as_posix() for path in files], "resolved_files": [describe_file(path) for path in files], "resolved_file_count": len(files), "test": describe_file(PROJECT_DIR / "tests/test_public_adaptive_scale097_neural_fixed_delta_diagnostic.py"), "config": describe_file(args.config.resolve()), "sidecar": describe_file(args.config.with_suffix(".sha256").resolve())}


def _snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    specs = {**config["lineage"], **config["input_identities"]}
    result: dict[str, Any] = {}
    for name, spec in specs.items():
        path = _verify(spec)
        result[name] = {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def _year_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01 01:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm")


def _segments() -> dict[str, pd.DatetimeIndex]:
    def rows(start: str, end: str) -> pd.DatetimeIndex:
        return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")
    return {"full": _year_index(), "H1": rows("2024-01-01 01:00", "2024-07-01 00:00"), "H2": rows("2024-07-01 01:00", "2025-01-01 00:00"), "Q1": rows("2024-01-01 01:00", "2024-04-01 00:00"), "Q2": rows("2024-04-01 01:00", "2024-07-01 00:00"), "Q3": rows("2024-07-01 01:00", "2024-10-01 00:00"), "Q4": rows("2024-10-01 01:00", "2025-01-01 00:00")}


def compose(A: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    if not A.index.equals(delta.index) or tuple(A.columns) != TARGETS or tuple(delta.columns) != TARGETS:
        raise AssertionError("A/delta alignment differs")
    if np.any(delta[IDENTITY].to_numpy(dtype=np.float64) != 0.0):
        raise AssertionError("G3 delta differs")
    candidate = A.copy()
    for group in ACTIVE:
        candidate[group] = np.clip(A[group].to_numpy(dtype=np.float64) + delta[group].to_numpy(dtype=np.float64), 0.0, 1.02 * CAPACITY[group])
    if not np.array_equal(candidate[IDENTITY].to_numpy().view(np.uint64), A[IDENTITY].to_numpy().view(np.uint64)):
        raise AssertionError("G3 identity differs")
    return candidate


def _group(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, float]:
    capacity = CAPACITY[group]
    y, p = actual.to_numpy(dtype=np.float64), prediction.to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & (y >= 0.10 * capacity)
    y, p = y[valid], p[valid]
    error = np.abs(p - y) / capacity
    n = 1.0 - float(error.mean())
    points = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    f = float(np.sum(y * points) / np.sum(y * 4.0))
    return {"total_score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f}


def _mixed(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, float]:
    values = [_group(actual[group], prediction[group], group) for group in TARGETS]
    n = float(np.mean([value["one_minus_nmae"] for value in values]))
    f = float(np.mean([value["ficr"] for value in values]))
    return {"total_score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f}


def diagnostic(actual: pd.DataFrame, A: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    all_positive = True
    for name, rows in _segments().items():
        groups: dict[str, Any] = {}
        for group in ACTIVE:
            before, after = _group(actual.loc[rows, group], A.loc[rows, group], group), _group(actual.loc[rows, group], candidate.loc[rows, group], group)
            delta = after["total_score"] - before["total_score"]
            groups[group] = {"A": before, "candidate": after, "delta_total_score": delta}
            all_positive = all_positive and delta > 0.0
        before_mixed, after_mixed = _mixed(actual.loc[rows], A.loc[rows]), _mixed(actual.loc[rows], candidate.loc[rows])
        mixed_delta = after_mixed["total_score"] - before_mixed["total_score"]
        comparisons[name] = {"groups": groups, "mixed_A": before_mixed, "mixed_candidate": after_mixed, "mixed_delta_total_score": mixed_delta, "mixed_delta_one_minus_nmae": after_mixed["one_minus_nmae"] - before_mixed["one_minus_nmae"], "mixed_delta_ficr": after_mixed["ficr"] - before_mixed["ficr"]}
        all_positive = all_positive and mixed_delta > 0.0
    full = comparisons["full"]
    components = bool(full["mixed_delta_one_minus_nmae"] >= 0.0 and full["mixed_delta_ficr"] >= 0.0)
    return {"comparison": "candidate minus A only", "comparisons": comparisons, "all_21_score_deltas_strictly_positive": all_positive, "full_components_nonnegative": components, "diagnostic_GO": bool(all_positive and components), "no_retune_retry_rescue": True}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.config = args.config.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    config = _verify_config(args.config)
    args.out_dir.mkdir(parents=True)
    _copy(args.config, args.out_dir / "preregister.json")
    _copy(args.config.with_suffix(".sha256"), args.out_dir / "preregister.sha256")
    closure, upstream = _closure(args), _snapshot(config)
    _write(args.out_dir / "before_candidate_and_metric_lock.json", {"schema_version": 1, "created_utc": utc_now(), "config_sha256": CONFIG_SHA, "new_candidate_cells_before_lock": 0, "new_metric_values_before_lock": 0, "closure": closure, "upstream": upstream})
    A = pd.read_parquet(_verify(config["input_identities"]["A_scale097_2024"])).astype(np.float64)
    delta = pd.read_parquet(_verify(config["input_identities"]["fixed_neural_primary_delta_2024"])).astype(np.float64)
    for frame in (A, delta):
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(_year_index()) or tuple(frame.columns) != TARGETS:
            raise AssertionError("input schema/index differs")
    changed = {group: int(np.count_nonzero(delta[group].to_numpy())) for group in ACTIVE}
    if changed != EXPECTED_DELTA_ROWS:
        raise AssertionError("fixed delta row counts differ")
    candidate = compose(A, delta)
    candidate_path = args.out_dir / "candidate_scale097_plus_fixed_neural_delta_2024.parquet"
    _atomic_parquet(candidate, candidate_path)
    labels = pd.read_csv(_verify(config["input_identities"]["labels"]))
    labels.index = pd.DatetimeIndex(pd.to_datetime(labels.pop("kst_dtm")), name="forecast_kst_dtm")
    result = diagnostic(labels.loc[_year_index(), list(TARGETS)].astype(np.float64), A, candidate)
    result.update({"fixed_delta_nonzero_rows": changed, "candidate": describe_file(candidate_path), "G3_A_bit_identity": True, "risk": config["risk_classification"]})
    _write(args.out_dir / "diagnostic_results.json", result)
    final = {"executed": False, "reason": "separate final prescore is allowed only after diagnostic GO" if result["diagnostic_GO"] else "diagnostic NO-GO", "final_refit_calls": 0, "2025_candidate_cells": 0, "CSV_created": False}
    _write(args.out_dir / "final_results.json", final)
    if closure != _closure(args) or upstream != _snapshot(config):
        raise AssertionError("source or upstream changed")
    files = sorted((path for path in args.out_dir.rglob("*") if path.is_file() and path.name != "manifest.json"), key=lambda path: path.relative_to(args.out_dir).as_posix())
    manifest = {"schema_version": 1, "artifact_type": "public_adaptive_scale097_plus_neural_fixed_delta_diagnostic_v1", "created_utc": utc_now(), "config_sha256": CONFIG_SHA, "risk": config["risk_classification"], "immutable_interaction": config["immutable_single_interaction"], "source_provenance": {"closure": closure, "locked_before_candidate_and_metric": True, "unchanged": True}, "upstream_before_after_exact": True, "diagnostic": result, "final": final, "outputs": [describe_file(path) for path in files], "output_count_excluding_manifest": len(files), "runtime": {"packages": package_versions()}, "git": git_state(PROJECT_DIR)}
    manifest_path = args.out_dir / "manifest.json"
    _write(manifest_path, manifest)
    print(f"diagnostic_GO={result['diagnostic_GO']}")
    print(f"manifest_sha256={sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()

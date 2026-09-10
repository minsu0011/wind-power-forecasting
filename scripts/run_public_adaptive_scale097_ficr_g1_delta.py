"""Build the frozen 0.97 plus direct-FICR G1-delta candidate."""

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

from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now, write_json_atomic  # noqa: E402


CONFIG_SHA = "30899a8496184f32de228edf154eab7764eb8c52144940f38e12db7d0c29fcb2"
TARGETS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
GROUP = "kpx_group_1"
IDENTITY = ("kpx_group_2", "kpx_group_3")
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
PROTECTED_UPSTREAM_ROOTS = (
    "artifacts/postgate/public_adaptive_scale097_g2_delta_v2",
    "artifacts/postgate/ficr_bayes_v4_composition",
    "artifacts/postgate/ficr_bayes_decision",
)
PROTECTED_UPSTREAM_FILES = (
    "artifacts/final_cf_fix/corrected_recent_v4.csv",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/public_adaptive_scale097_ficr_g1_delta_preregister_v1.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/postgate/public_adaptive_scale097_ficr_g1_delta_v1"))
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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
        frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.6f", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _path(spec: Mapping[str, Any]) -> Path:
    path = Path(str(spec["path"]))
    return path if path.is_absolute() else PROJECT_DIR / path


def _verify(spec: Mapping[str, Any]) -> Path:
    path = _path(spec)
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
        raise AssertionError("config sidecar differs")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["immutable_single_candidate"]["smoothed_variant_forbidden"] is not True:
        raise AssertionError("smoothed prohibition differs")
    if config["risk_classification"]["selection_unsafe"] is not True:
        raise AssertionError("risk differs")
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
    queue = [entry.resolve()]
    seen: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        if PROJECT_DIR not in path.parents:
            raise AssertionError("closure escapes project")
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    hits = _local(alias.name)
                    if hits:
                        queue.extend(hits)
                    elif (PROJECT_DIR / alias.name.split(".")[0]).exists():
                        raise AssertionError(f"unresolved local import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                hits = _local(base)
                if hits:
                    queue.extend(hits)
                elif base and (PROJECT_DIR / base.split(".")[0]).exists():
                    raise AssertionError(f"unresolved local from-import: {base}")
                for alias in node.names:
                    if alias.name != "*":
                        queue.extend(_local(f"{base}.{alias.name}"))
    return tuple(sorted(seen))


def _closure(args: argparse.Namespace) -> dict[str, Any]:
    files = resolve_ast_closure(Path(__file__))
    return {
        "resolver": "recursive Python AST local imports plus package initializers",
        "resolved_relative_paths": [path.relative_to(PROJECT_DIR).as_posix() for path in files],
        "resolved_files": [describe_file(path) for path in files],
        "resolved_file_count": len(files),
        "unresolved_local_imports": [],
        "test": describe_file(PROJECT_DIR / "tests/test_public_adaptive_scale097_ficr_g1_delta.py"),
        "config": describe_file(args.config.resolve()),
        "sidecar": describe_file(args.config.with_suffix(".sha256").resolve()),
    }


def protected_upstream_snapshot() -> dict[str, Any]:
    """Hash every protected upstream byte before and after this run."""

    files: set[Path] = set()
    for relative in PROTECTED_UPSTREAM_ROOTS:
        root = PROJECT_DIR / relative
        if not root.is_dir():
            raise FileNotFoundError(root)
        files.update(path.resolve() for path in root.rglob("*") if path.is_file())
    for relative in PROTECTED_UPSTREAM_FILES:
        path = (PROJECT_DIR / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        files.add(path)
    records = [
        {
            "relative_path": path.relative_to(PROJECT_DIR).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]
    return {
        "schema_version": 1,
        "protected_roots": list(PROTECTED_UPSTREAM_ROOTS),
        "protected_files": list(PROTECTED_UPSTREAM_FILES),
        "file_count": len(records),
        "files": records,
    }


def _year_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01 01:00", f"{year + 1}-01-01 00:00", freq="h", name="forecast_kst_dtm")


def _segments(year: int) -> dict[str, pd.DatetimeIndex]:
    def interval(start: str, end: str) -> pd.DatetimeIndex:
        return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")
    return {
        "full": _year_index(year),
        "H1": interval(f"{year}-01-01 01:00", f"{year}-07-01 00:00"),
        "H2": interval(f"{year}-07-01 01:00", f"{year + 1}-01-01 00:00"),
        "Q1": interval(f"{year}-01-01 01:00", f"{year}-04-01 00:00"),
        "Q2": interval(f"{year}-04-01 01:00", f"{year}-07-01 00:00"),
        "Q3": interval(f"{year}-07-01 01:00", f"{year}-10-01 00:00"),
        "Q4": interval(f"{year}-10-01 01:00", f"{year + 1}-01-01 00:00"),
    }


def _read_parquet(spec: Mapping[str, Any], year: int) -> pd.DataFrame:
    frame = pd.read_parquet(_verify(spec)).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGETS or not frame.index.equals(_year_index(year)):
        raise AssertionError("prediction schema/index differs")
    return frame


def _read_submission_prediction(spec: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    submission = pd.read_csv(_verify(spec), encoding="utf-8-sig", dtype={"forecast_id": "string", "forecast_kst_dtm": "string"})
    if tuple(submission.columns) != ("forecast_id", "forecast_kst_dtm", *TARGETS) or len(submission) != 8760:
        raise AssertionError("reference CSV schema/rows differs")
    index = pd.DatetimeIndex(pd.to_datetime(submission["forecast_kst_dtm"]), name="forecast_kst_dtm")
    prediction = submission.loc[:, list(TARGETS)].astype(np.float64)
    prediction.index = index
    return submission, prediction


def _bits(series: pd.Series) -> np.ndarray:
    return np.ascontiguousarray(series.to_numpy(dtype=np.float64)).view(np.uint64)


def compose(A: pd.DataFrame, reference: pd.DataFrame, adjusted: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not A.index.equals(reference.index) or not A.index.equals(adjusted.index):
        raise AssertionError("composition index differs")
    delta = adjusted - reference
    for group in IDENTITY:
        if np.any(delta[group].to_numpy() != 0.0):
            raise AssertionError(f"delta not G1-only: {group}")
    candidate = A.copy()
    for group in TARGETS:
        candidate[group] = np.clip(A[group].to_numpy() + delta[group].to_numpy(), 0.0, 1.02 * CAPACITY[group])
    for group in IDENTITY:
        if not np.array_equal(_bits(candidate[group]), _bits(A[group])):
            raise AssertionError(f"candidate identity differs: {group}")
    return candidate, delta


def _group(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, float]:
    capacity = CAPACITY[group]
    y = actual.to_numpy(dtype=np.float64)
    p = prediction.to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & (y >= 0.10 * capacity)
    y, p = y[valid], p[valid]
    error = np.abs(p - y) / capacity
    n = 1.0 - float(error.mean())
    price = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    f = float(np.sum(y * price) / np.sum(y * 4.0))
    return {"total_score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f}


def _mixed(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, float]:
    rows = [_group(actual[group], prediction[group], group) for group in TARGETS]
    n = float(np.mean([row["one_minus_nmae"] for row in rows]))
    f = float(np.mean([row["ficr"] for row in rows]))
    return {"total_score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f}


def _labels(spec: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(_verify(spec))
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm")), name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGETS or len(frame) != int(spec["rows"]):
        raise AssertionError("labels differ")
    return frame.astype(np.float64)


def _diagnostic(config: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    inputs = config["input_identities"]
    A = _read_parquet(inputs["A_2024_scale097"], 2024)
    reference = _read_parquet(inputs["ficr_delta_reference_2024"], 2024)
    adjusted = _read_parquet(inputs["ficr_delta_adjusted_2024"], 2024)
    candidate, delta = compose(A, reference, adjusted)
    labels = _labels(inputs["labels"])
    comparisons: dict[str, Any] = {}
    all_positive = True
    for name, rows in _segments(2024).items():
        ga, gb = _group(labels.loc[rows, GROUP], A.loc[rows, GROUP], GROUP), _group(labels.loc[rows, GROUP], candidate.loc[rows, GROUP], GROUP)
        ma, mb = _mixed(labels.loc[rows], A.loc[rows]), _mixed(labels.loc[rows], candidate.loc[rows])
        gd, md = gb["total_score"] - ga["total_score"], mb["total_score"] - ma["total_score"]
        comparisons[name] = {"G1_A": ga, "G1_candidate": gb, "G1_delta_total_score": gd, "mixed_A": ma, "mixed_candidate": mb, "mixed_delta_total_score": md, "mixed_delta_one_minus_nmae": mb["one_minus_nmae"] - ma["one_minus_nmae"], "mixed_delta_ficr": mb["ficr"] - ma["ficr"]}
        all_positive = all_positive and gd > 0 and md > 0
    full = comparisons["full"]
    go = bool(all_positive and full["mixed_delta_one_minus_nmae"] >= 0 and full["mixed_delta_ficr"] >= 0)
    A_path = out_dir / config["output_contract"]["diagnostic_A"]
    candidate_path = out_dir / config["output_contract"]["diagnostic_candidate"]
    delta_path = out_dir / "diagnostic_2024/fixed_ficr_g1_delta.parquet"
    _atomic_parquet(A, A_path)
    _atomic_parquet(candidate, candidate_path)
    _atomic_parquet(delta, delta_path)
    return {"comparison": "candidate minus A only", "comparisons": comparisons, "diagnostic_GO": go, "recommendation": "candidate_over_A" if go else "A_over_candidate", "G1_delta_nonzero_rows": int(np.count_nonzero(delta[GROUP].to_numpy())), "G2_G3_identity": True, "outputs": [describe_file(path) for path in (A_path, candidate_path, delta_path)], "no_retuning": True}


def _validate_csv(path: Path, sample: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    if not path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise AssertionError("CSV BOM differs")
    text = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    if tuple(text.columns) != tuple(sample.columns) or len(text) != 8760 or not text[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]].astype("string")):
        raise AssertionError("CSV schema/sample differs")
    for group in TARGETS:
        expected = prediction[group].reset_index(drop=True).map(lambda value: f"{float(value):.6f}").astype("string")
        if not text[group].equals(expected):
            raise AssertionError(f"CSV roundtrip differs: {group}")
    values = text.loc[:, list(TARGETS)].astype(np.float64).to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError("CSV non-finite")
    for pos, group in enumerate(TARGETS):
        if values[:, pos].min() < 0 or values[:, pos].max() > 1.02 * CAPACITY[group] + 5e-7:
            raise AssertionError("CSV bounds differs")
    return {"rows": 8760, "schema_sample_time": True, "BOM": True, "six_decimals": True, "finite_bounds": True, "sha256": sha256_file(path)}


def _final(config: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    inputs = config["input_identities"]
    A = _read_parquet(inputs["A_final_scale097"], 2025)
    _, reference = _read_submission_prediction(inputs["ficr_delta_reference_final_csv"])
    adjusted = _read_parquet(inputs["ficr_composed_final_direct"], 2025)
    candidate, delta = compose(A, reference, adjusted)
    prediction_path = out_dir / config["output_contract"]["final_prediction"]
    delta_path = out_dir / "final/fixed_ficr_g1_delta.parquet"
    _atomic_parquet(candidate, prediction_path)
    _atomic_parquet(delta, delta_path)
    saved = pd.read_parquet(prediction_path).astype(np.float64)
    for group in TARGETS:
        if not np.array_equal(_bits(saved[group]), _bits(candidate[group])):
            raise AssertionError(f"prediction roundtrip differs: {group}")
    sample = pd.read_csv(_verify(inputs["sample"]), encoding="utf-8-sig", dtype={"forecast_id": "string", "forecast_kst_dtm": "string"})
    if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGETS) or len(sample) != 8760:
        raise AssertionError("sample differs")
    if not pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"]), name="forecast_kst_dtm").equals(_year_index(2025)):
        raise AssertionError("sample time differs")
    submission = sample.copy()
    for group in TARGETS:
        submission[group] = candidate[group].to_numpy()
    csv_path = out_dir / config["output_contract"]["CSV"]
    _atomic_csv(submission, csv_path)
    return {"prediction": describe_file(prediction_path), "delta": describe_file(delta_path), "submission": describe_file(csv_path), "validation": _validate_csv(csv_path, sample, candidate), "G1_delta_nonzero_rows": int(np.count_nonzero(delta[GROUP].to_numpy())), "G2_G3_A_bit_identity": True, "formula_exact": True}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = _verify_config(args.config)
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    _copy(args.config, args.out_dir / "preregister.json")
    _copy(args.config.with_suffix(".sha256"), args.out_dir / "preregister.sha256")
    closure = _closure(args)
    lock_path = args.out_dir / "source_closure_before_metric_and_materialization_lock.json"
    _write_json(lock_path, {"schema_version": 1, "created_utc": utc_now(), "config_sha256": CONFIG_SHA, "new_metric_values_before_lock": 0, "new_candidate_values_before_lock": 0, "new_CSV_bytes_before_lock": 0, "closure": closure})
    upstream_before = protected_upstream_snapshot()
    _write_json(
        args.out_dir / "protected_upstream_snapshot_before_metric_and_CSV.json",
        {**upstream_before, "captured_utc": utc_now(), "phase": "before_metric_and_CSV"},
    )
    diagnostic = _diagnostic(config, args.out_dir)
    _write_json(args.out_dir / "interaction_diagnostic_2024.json", diagnostic)
    final = _final(config, args.out_dir)
    _write_json(args.out_dir / "final_results.json", {"schema_version": 1, "created_utc": utc_now(), "risk": config["risk_classification"], "diagnostic_recommendation": diagnostic["recommendation"], "final": final})
    upstream_after = protected_upstream_snapshot()
    if upstream_before != upstream_after:
        raise AssertionError("protected upstream bytes changed during execution")
    _write_json(
        args.out_dir / "protected_upstream_snapshot_after_metric_and_CSV.json",
        {
            **upstream_after,
            "captured_utc": utc_now(),
            "phase": "after_metric_and_CSV",
            "exactly_equal_to_before": True,
        },
    )
    if closure != _closure(args):
        raise AssertionError("source closure changed")
    files = sorted((path for path in args.out_dir.rglob("*") if path.is_file() and path.name != "manifest.json"), key=lambda path: path.relative_to(args.out_dir).as_posix())
    manifest = {"schema_version": 1, "artifact_type": "public_adaptive_scale097_plus_fixed_direct_ficr_g1_delta_v1", "created_utc": utc_now(), "config_sha256": CONFIG_SHA, "risk": config["risk_classification"], "source_provenance": {"recursive_AST_closure": closure, "locked_before_metric_and_materialization": True, "unchanged_before_manifest": True}, "protected_upstream_nonmutation": {"before_after_exact": True, "file_count": upstream_before["file_count"], "roots": upstream_before["protected_roots"], "files": upstream_before["protected_files"]}, "immutable_candidate": config["immutable_single_candidate"], "deduplication": config["deduplication"], "diagnostic": diagnostic, "final": final, "outputs": [describe_file(path) for path in files], "output_count_excluding_manifest": len(files), "runtime": {"packages": package_versions()}, "git": git_state(PROJECT_DIR)}
    _write_json(args.out_dir / "manifest.json", manifest)
    print(f"diagnostic_GO={diagnostic['diagnostic_GO']} recommendation={diagnostic['recommendation']}")
    print(f"CSV_sha256={final['submission']['sha256']}")
    print(f"manifest_sha256={sha256_file(args.out_dir / 'manifest.json')}")


if __name__ == "__main__":
    main()

"""Evaluate the frozen 0.97 within-run smoothing candidate and gate 2025 output."""

from __future__ import annotations

import argparse
import ast
import hashlib
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
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402
from src.temporal import smooth_within_runs  # noqa: E402


CONFIG_SHA = "ed4c06d42b7e48b894ad6cfc90766071f0857d6a6e2c03edf32f7b4fd9fadc80"
FACTOR = np.float64(0.97)
STRENGTHS = {"kpx_group_1": 0.15, "kpx_group_2": 0.05, "kpx_group_3": 0.0}
TARGETS = tuple(TARGET_COLS)
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/public_scale097_within_run_smoothing_preregister_v1.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/public_scale097_within_run_smoothing_v1"),
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


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def resolve_ast_closure(entrypoint: Path) -> tuple[Path, ...]:
    pending = [entrypoint.resolve()]
    seen: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        if PROJECT_DIR not in current.parents:
            raise AssertionError("closure path escapes project")
        seen.add(current)
        tree = ast.parse(current.read_text(encoding="utf-8"), filename=str(current))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    hits = _module_files(alias.name)
                    if hits:
                        pending.extend(hits)
                    elif (PROJECT_DIR / alias.name.split(".")[0]).exists():
                        raise AssertionError(f"unresolved local import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                hits = _module_files(base)
                if hits:
                    pending.extend(hits)
                elif base and (PROJECT_DIR / base.split(".")[0]).exists():
                    raise AssertionError(f"unresolved local from-import: {base}")
                for alias in node.names:
                    if alias.name != "*":
                        pending.extend(_module_files(f"{base}.{alias.name}"))
    return tuple(sorted(seen))


def _closure(args: argparse.Namespace) -> dict[str, Any]:
    files = resolve_ast_closure(Path(__file__))
    test = PROJECT_DIR / "tests/test_public_scale097_within_run_smoothing.py"
    return {
        "resolver": "recursive Python AST local imports plus package initializers",
        "entrypoint": Path(__file__).resolve().relative_to(PROJECT_DIR).as_posix(),
        "resolved_relative_paths": [path.relative_to(PROJECT_DIR).as_posix() for path in files],
        "resolved_files": [describe_file(path) for path in files],
        "resolved_file_count": len(files),
        "unresolved_local_imports": [],
        "test": describe_file(test),
        "config": describe_file(args.config.resolve()),
        "config_sidecar": describe_file(args.config.with_suffix(".sha256").resolve()),
    }


def _year_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{year}-01-01 01:00", f"{year + 1}-01-01 00:00", freq="h", name="forecast_kst_dtm"
    )


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


def _read_prediction(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index) or tuple(frame.columns) != TARGETS:
        raise AssertionError(f"prediction schema/index differs: {path}")
    if not np.isfinite(frame.to_numpy()).all():
        raise AssertionError(f"prediction contains non-finite values: {path}")
    return frame


def _read_labels(path: Path, expected_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string")
    if "kst_dtm" not in frame:
        raise AssertionError("label timestamp column missing")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm")), name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGETS or len(frame) != expected_rows:
        raise AssertionError("label schema/rows differs")
    for group in TARGETS:
        frame[group] = pd.to_numeric(frame[group], errors="coerce")
    return frame.astype(np.float64)


def _bits(series: pd.Series) -> np.ndarray:
    return np.ascontiguousarray(series.to_numpy(dtype=np.float64)).view(np.uint64)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _frame_value_sha(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(np.ascontiguousarray(frame.index.view("i8")).tobytes())
    digest.update(np.ascontiguousarray(frame.to_numpy(dtype=np.float64)).tobytes())
    return digest.hexdigest()


def _candidate(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaled = base.copy()
    for group in TARGETS:
        bound = np.float64(1.02 * CAPACITY_KWH[group])
        scaled[group] = np.clip(FACTOR * base[group].to_numpy(dtype=np.float64), 0.0, bound)
    candidate = smooth_within_runs(scaled, STRENGTHS)
    for group in TARGETS:
        candidate[group] = np.clip(
            candidate[group].to_numpy(dtype=np.float64), 0.0, np.float64(1.02 * CAPACITY_KWH[group])
        )
    if not np.array_equal(_bits(candidate["kpx_group_3"]), _bits(scaled["kpx_group_3"])):
        raise AssertionError("G3 identity differs")
    return scaled, candidate


def _mask_sha(rows: pd.DatetimeIndex, full_index: pd.DatetimeIndex) -> str:
    mask = full_index.isin(rows).astype(np.uint8)
    return _sha_bytes(np.ascontiguousarray(mask).tobytes())


def _group_metrics(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, float | int]:
    capacity = float(CAPACITY_KWH[group])
    y = actual.to_numpy(dtype=np.float64)
    p = prediction.to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & (y >= 0.10 * capacity)
    y, p = y[valid], p[valid]
    if y.size == 0:
        raise AssertionError(f"no valid metric rows: {group}")
    error = np.abs(p - y) / capacity
    one_minus_nmae = 1.0 - float(error.mean())
    price = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    ficr = float(np.sum(y * price) / np.sum(y * 4.0))
    return {
        "n_evaluated": int(y.size),
        "total_score": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def _mixed(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, float]:
    records = [_group_metrics(actual[group], prediction[group], group) for group in TARGETS]
    n = float(np.mean([float(item["one_minus_nmae"]) for item in records]))
    f = float(np.mean([float(item["ficr"]) for item in records]))
    return {"total_score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f}


def _diagnostic_one(
    labels: pd.DataFrame,
    scaled: pd.DataFrame,
    candidate: pd.DataFrame,
    segments: Mapping[str, pd.DatetimeIndex],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    all_active_positive = True
    all_mixed_positive = True
    for segment in SEGMENTS:
        rows = segments[segment]
        group_records: dict[str, Any] = {}
        for group in ("kpx_group_1", "kpx_group_2"):
            before = _group_metrics(labels.loc[rows, group], scaled.loc[rows, group], group)
            after = _group_metrics(labels.loc[rows, group], candidate.loc[rows, group], group)
            delta = float(after["total_score"]) - float(before["total_score"])
            group_records[group] = {"baseline": before, "candidate": after, "delta_total_score": delta}
            all_active_positive = all_active_positive and delta > 0.0
        mixed_before = _mixed(labels.loc[rows], scaled.loc[rows])
        mixed_after = _mixed(labels.loc[rows], candidate.loc[rows])
        mixed_delta = mixed_after["total_score"] - mixed_before["total_score"]
        all_mixed_positive = all_mixed_positive and mixed_delta > 0.0
        comparisons[segment] = {
            "groups": group_records,
            "mixed": {
                "baseline": mixed_before,
                "candidate": mixed_after,
                "delta_total_score": mixed_delta,
                "delta_one_minus_nmae": mixed_after["one_minus_nmae"] - mixed_before["one_minus_nmae"],
                "delta_ficr": mixed_after["ficr"] - mixed_before["ficr"],
            },
        }
    full = comparisons["full"]["mixed"]
    full_components_nonnegative = bool(
        full["delta_one_minus_nmae"] >= 0.0 and full["delta_ficr"] >= 0.0
    )
    return {
        "comparisons": comparisons,
        "active_group_check_count": 14,
        "mixed_check_count": 7,
        "all_active_group_deltas_strictly_positive": bool(all_active_positive),
        "all_mixed_deltas_strictly_positive": bool(all_mixed_positive),
        "full_mixed_components_nonnegative": full_components_nonnegative,
        "passed": bool(all_active_positive and all_mixed_positive and full_components_nonnegative),
    }


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


def _validate_csv(path: Path, sample: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    if not path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise AssertionError("CSV BOM differs")
    text = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    if tuple(text.columns) != tuple(sample.columns) or len(text) != 8760:
        raise AssertionError("CSV schema/rows differs")
    if not text[["forecast_id", "forecast_kst_dtm"]].equals(
        sample[["forecast_id", "forecast_kst_dtm"]].astype("string")
    ):
        raise AssertionError("CSV sample identity differs")
    for group in TARGETS:
        expected = prediction[group].reset_index(drop=True).map(lambda value: f"{float(value):.6f}").astype("string")
        if not text[group].equals(expected):
            raise AssertionError(f"CSV roundtrip differs: {group}")
    values = text.loc[:, list(TARGETS)].astype(np.float64).to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError("CSV contains non-finite values")
    for position, group in enumerate(TARGETS):
        if values[:, position].min() < 0.0 or values[:, position].max() > 1.02 * CAPACITY_KWH[group] + 5e-7:
            raise AssertionError(f"CSV bounds differ: {group}")
    return {
        "rows": 8760,
        "schema_sample_id_time": True,
        "utf8_BOM": True,
        "six_decimals": True,
        "finite_bounds": True,
        "sha256": sha256_file(path),
    }


def _final(config: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    specs = config["input_identities"]
    index = _year_index(2025)
    plain = _read_prediction(_verify(specs["final_plain_scale097_prediction_gate_deferred"]), index)
    candidate = smooth_within_runs(plain, STRENGTHS)
    for group in TARGETS:
        candidate[group] = np.clip(
            candidate[group].to_numpy(dtype=np.float64), 0.0, np.float64(1.02 * CAPACITY_KWH[group])
        )
    if not np.array_equal(_bits(candidate["kpx_group_3"]), _bits(plain["kpx_group_3"])):
        raise AssertionError("2025 G3 identity differs")
    prediction_path = out_dir / config["output_contract"]["prediction_if_pass"]
    _atomic_parquet(candidate, prediction_path)
    saved = pd.read_parquet(prediction_path).astype(np.float64)
    for group in TARGETS:
        if not np.array_equal(_bits(candidate[group]), _bits(saved[group])):
            raise AssertionError(f"parquet bit roundtrip differs: {group}")
    sample = pd.read_csv(
        _verify(specs["sample_gate_deferred"]),
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGETS) or len(sample) != 8760:
        raise AssertionError("sample schema/rows differs")
    if not pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"]), name="forecast_kst_dtm").equals(index):
        raise AssertionError("sample time differs")
    submission = sample.copy()
    for group in TARGETS:
        submission[group] = candidate[group].to_numpy(dtype=np.float64)
    csv_path = out_dir / config["output_contract"]["csv_if_pass"]
    _atomic_csv(submission, csv_path)
    return {
        "plain_input": describe_file(_verify(specs["final_plain_scale097_prediction_gate_deferred"])),
        "plain_csv_lineage": describe_file(_verify(specs["final_plain_scale097_csv_lineage_gate_deferred"])),
        "prediction": describe_file(prediction_path),
        "csv": describe_file(csv_path),
        "csv_validation": _validate_csv(csv_path, sample, candidate),
        "G3_plain_candidate_float64_bits_exact": True,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if sha256_file(args.config) != CONFIG_SHA:
        raise AssertionError("config hash differs")
    expected_sidecar = f"{CONFIG_SHA}  {args.config.name}\n"
    if args.config.with_suffix(".sha256").read_text(encoding="utf-8") != expected_sidecar:
        raise AssertionError("config sidecar differs")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["immutable_candidate"]["strengths"] != STRENGTHS or float(config["immutable_candidate"]["factor"]) != float(FACTOR):
        raise AssertionError("candidate literals differ")
    if config["risk_classification"]["selection_unsafe"] is not True:
        raise AssertionError("risk disclosure differs")
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    _copy(args.config, args.out_dir / "preregister.json")
    _copy(args.config.with_suffix(".sha256"), args.out_dir / "preregister.sha256")

    closure = _closure(args)
    _write_json(
        args.out_dir / "source_closure_before_candidate_and_labels.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "prediction_value_cells_read": 0,
            "candidate_value_cells_materialized": 0,
            "label_value_cells_read": 0,
            "metric_values_computed": 0,
            "closure": closure,
        },
    )

    specs = config["input_identities"]
    index = _year_index(2024)
    segments = _segments(2024)
    prepared: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, spec_name in (
        ("primary", "primary_locked_v3_2024"),
        ("interaction", "interaction_recent_v4_2024"),
    ):
        base = _read_prediction(_verify(specs[spec_name]), index)
        prepared[name] = _candidate(base)
    mask_hashes = {name: _mask_sha(rows, index) for name, rows in segments.items()}
    candidate_hashes = {
        name: {"scaled_baseline": _frame_value_sha(pair[0]), "candidate": _frame_value_sha(pair[1])}
        for name, pair in prepared.items()
    }
    lock_path = args.out_dir / config["output_contract"]["prescore_lock"]
    lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA,
        "candidate_count": 1,
        "factor": float(FACTOR),
        "strengths": STRENGTHS,
        "candidate_value_hashes": candidate_hashes,
        "mask_hashes": mask_hashes,
        "G3_identity_primary": True,
        "G3_identity_interaction": True,
        "2024_label_physical_bytes_hashed_before_lock": 0,
        "2024_label_value_cells_read": 0,
        "metric_values_computed": 0,
    }
    _write_json(lock_path, lock)
    for name, pair in prepared.items():
        if _frame_value_sha(pair[0]) != candidate_hashes[name]["scaled_baseline"] or _frame_value_sha(pair[1]) != candidate_hashes[name]["candidate"]:
            raise AssertionError("candidate hash changed before labels")
    if {name: _mask_sha(rows, index) for name, rows in segments.items()} != mask_hashes:
        raise AssertionError("mask hash changed before labels")

    label_spec = specs["labels_2022_2024_physical_only_before_lock"]
    labels = _read_labels(_verify(label_spec), int(label_spec["rows"])).loc[index]
    diagnostics = {
        name: _diagnostic_one(labels, pair[0], pair[1], segments) for name, pair in prepared.items()
    }
    passed = bool(all(item["passed"] for item in diagnostics.values()))
    stage1 = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA,
        "candidate_mask_prescore_lock": describe_file(lock_path),
        "diagnostics": diagnostics,
        "passed": passed,
        "decision": "GO_materialize_2025" if passed else "REJECT_no_2025_read_or_output",
    }
    _write_json(args.out_dir / config["output_contract"]["stage1_results"], stage1)

    final = _final(config, args.out_dir) if passed else None
    final_results = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "selection_unsafe": True,
        "multiple_testing": True,
        "strict_final_isolation": False,
        "leaderboard_score_claim": False,
        "stage1_passed": passed,
        "2025_prediction_value_cells_read": 8760 * 3 if passed else 0,
        "2025_csv_created": bool(passed),
        "final": final,
    }
    _write_json(args.out_dir / config["output_contract"]["final_results"], final_results)

    closure_after = _closure(args)
    if closure_after != closure:
        raise AssertionError("source closure changed")
    for spec_name in ("primary_locked_v3_2024", "interaction_recent_v4_2024", "labels_2022_2024_physical_only_before_lock", "temporal_implementation"):
        _verify(specs[spec_name])
    if passed:
        for spec_name in ("final_plain_scale097_prediction_gate_deferred", "final_plain_scale097_csv_lineage_gate_deferred", "sample_gate_deferred"):
            _verify(specs[spec_name])
    files = sorted(
        (path for path in args.out_dir.rglob("*") if path.is_file() and path.name != "manifest.json"),
        key=lambda path: path.relative_to(args.out_dir).as_posix(),
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "public_scale097_within_run_smoothing_v1",
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA,
        "risk": config["risk_classification"],
        "source_provenance": {
            "closure": closure,
            "locked_before_candidate_and_labels": True,
            "unchanged_after_execution": True,
        },
        "candidate_mask_prescore_lock": describe_file(lock_path),
        "stage1": stage1,
        "final": final_results,
        "upstream_nonmutation_verified": True,
        "outputs": [describe_file(path) for path in files],
        "output_count_excluding_manifest": len(files),
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
    }
    _write_json(args.out_dir / config["output_contract"]["manifest"], manifest)
    print(f"stage1_passed={passed}", flush=True)
    if final is not None:
        print(f"csv_sha256={final['csv']['sha256']}", flush=True)
    print(f"manifest_sha256={sha256_file(args.out_dir / config['output_contract']['manifest'])}", flush=True)


if __name__ == "__main__":
    main()

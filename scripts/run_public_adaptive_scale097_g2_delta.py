"""Materialize the frozen Public-adaptive 0.97 plain and G2-delta candidates."""

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


CONFIG_SHA = "9ed9bc8d4ec73f714cbd4d1b1c41ac8b5c45a7812a1bd2bbe28a8c5b5b24c475"
V1_SHA = "5c637901229dbe3cde68a2c48123e3cf41acb86f88ec7c3db44e3f8322de459b"
FACTOR = np.float64(0.97)
TARGETS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
GROUP = "kpx_group_2"
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/public_adaptive_scale097_g2_delta_preregister_v2.json"),
    )
    parser.add_argument(
        "--v1-config", type=Path,
        default=Path("configs/public_adaptive_scale097_g2_delta_preregister_v1.json"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("artifacts/postgate/public_adaptive_scale097_g2_delta_v2"),
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
        frame.to_csv(
            temporary, index=False, encoding="utf-8-sig", float_format="%.6f",
            lineterminator="\n",
        )
        os.replace(temporary, path)
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


def _verify_configs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(args.config) != CONFIG_SHA:
        raise AssertionError("v2 config hash differs")
    if args.config.with_suffix(".sha256").read_text(encoding="utf-8") != f"{CONFIG_SHA}  {args.config.name}\n":
        raise AssertionError("v2 sidecar differs")
    if sha256_file(args.v1_config) != V1_SHA:
        raise AssertionError("v1 config hash differs")
    v2 = json.loads(args.config.read_text(encoding="utf-8"))
    v1 = json.loads(args.v1_config.read_text(encoding="utf-8"))
    if float(v2["immutable_candidates_repeated"]["factor"]) != float(FACTOR):
        raise AssertionError("factor differs")
    if v2["risk_classification"]["public_adaptive"] is not True:
        raise AssertionError("risk disclosure differs")
    _verify(v2["current_v5_independent_audit"])
    for spec in v1["lineage"].values():
        if isinstance(spec, Mapping) and "path" in spec:
            # The local audit was superseded at the same path after v1 freeze and
            # is deliberately replaced by v2's current independent audit record.
            if spec.get("sha256") != "60be7f71dc9f02fdfd2195bff845e73d53e66dc662992db5620178496e5b9613":
                _verify(spec)
    for spec in v1["input_identities"].values():
        _verify(spec)
    return v2, v1


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
    test = PROJECT_DIR / "tests/test_public_adaptive_scale097_g2_delta.py"
    return {
        "resolver": "recursive Python AST local imports plus package initializers",
        "entrypoint": Path(__file__).resolve().relative_to(PROJECT_DIR).as_posix(),
        "resolved_relative_paths": [path.relative_to(PROJECT_DIR).as_posix() for path in files],
        "resolved_files": [describe_file(path) for path in files],
        "resolved_file_count": len(files),
        "unresolved_local_imports": [],
        "test": describe_file(test),
        "v2_config": describe_file(args.config.resolve()),
        "v2_sidecar": describe_file(args.config.with_suffix(".sha256").resolve()),
        "v1_config": describe_file(args.v1_config.resolve()),
        "v1_sidecar": describe_file(args.v1_config.with_suffix(".sha256").resolve()),
    }


def _assert_closure(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    if first != second:
        raise AssertionError("AST source/config/test closure changed")


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


def _read_prediction(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index) or tuple(frame.columns) != TARGETS:
        raise AssertionError(f"prediction schema/index differs: {path}")
    if not np.isfinite(frame.to_numpy()).all():
        raise AssertionError("prediction contains non-finite values")
    return frame


def _bits(series: pd.Series) -> np.ndarray:
    return np.ascontiguousarray(series.to_numpy(dtype=np.float64)).view(np.uint64)


def candidates(base: pd.DataFrame, rescue: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not base.index.equals(rescue.index) or tuple(base.columns) != TARGETS or tuple(rescue.columns) != TARGETS:
        raise AssertionError("base/rescue alignment differs")
    delta = rescue - base
    for group in ("kpx_group_1", "kpx_group_3"):
        if not np.array_equal(_bits(rescue[group]), _bits(base[group])):
            raise AssertionError(f"frozen rescue identity differs: {group}")
        if np.any(delta[group].to_numpy() != 0.0):
            raise AssertionError(f"frozen delta is not zero: {group}")
    A = base.copy()
    B = base.copy()
    for group in TARGETS:
        bound = np.float64(1.02 * CAPACITY[group])
        a = np.clip(FACTOR * base[group].to_numpy(dtype=np.float64), 0.0, bound)
        b = np.clip(a + delta[group].to_numpy(dtype=np.float64), 0.0, bound)
        A[group] = a
        B[group] = b
    for group in ("kpx_group_1", "kpx_group_3"):
        if not np.array_equal(_bits(A[group]), _bits(B[group])):
            raise AssertionError(f"A/B identity group differs: {group}")
    return A, B, delta


def _group_metrics(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, float]:
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
    records = [_group_metrics(actual[group], prediction[group], group) for group in TARGETS]
    n = float(np.mean([item["one_minus_nmae"] for item in records]))
    f = float(np.mean([item["ficr"] for item in records]))
    return {"total_score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f}


def _read_labels(spec: Mapping[str, Any]) -> pd.DataFrame:
    path = _verify(spec)
    frame = pd.read_csv(path)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm")), name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGETS or len(frame) != int(spec["rows"]):
        raise AssertionError("label schema/rows differs")
    return frame.astype(np.float64)


def _diagnostic(v1: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    specs = v1["input_identities"]
    index = _year_index(2024)
    base = _read_prediction(_verify(specs["interaction_recent_v4_2024"]), index)
    rescue = _read_prediction(_verify(specs["interaction_g2_rescue_2024"]), index)
    A, B, delta = candidates(base, rescue)
    labels = _read_labels(specs["labels"])
    segments = _segments(2024)
    comparisons: dict[str, Any] = {}
    all_positive = True
    for name in SEGMENTS:
        rows = segments[name]
        ga = _group_metrics(labels.loc[rows, GROUP], A.loc[rows, GROUP], GROUP)
        gb = _group_metrics(labels.loc[rows, GROUP], B.loc[rows, GROUP], GROUP)
        ma = _mixed(labels.loc[rows], A.loc[rows])
        mb = _mixed(labels.loc[rows], B.loc[rows])
        g_delta = gb["total_score"] - ga["total_score"]
        m_delta = mb["total_score"] - ma["total_score"]
        comparisons[name] = {
            "G2_A": ga, "G2_B": gb, "G2_delta_total_score": g_delta,
            "mixed_A": ma, "mixed_B": mb, "mixed_delta_total_score": m_delta,
            "mixed_delta_one_minus_nmae": mb["one_minus_nmae"] - ma["one_minus_nmae"],
            "mixed_delta_ficr": mb["ficr"] - ma["ficr"],
        }
        all_positive = all_positive and g_delta > 0 and m_delta > 0
    full = comparisons["full"]
    components = full["mixed_delta_one_minus_nmae"] >= 0 and full["mixed_delta_ficr"] >= 0
    diagnostic_go = bool(all_positive and components)
    paths = {
        "A": out_dir / "diagnostic_2024/A_scale097.parquet",
        "B": out_dir / "diagnostic_2024/B_scale097_plus_g2_delta.parquet",
        "delta": out_dir / "diagnostic_2024/frozen_rescue_delta.parquet",
    }
    _atomic_parquet(A, paths["A"])
    _atomic_parquet(B, paths["B"])
    _atomic_parquet(delta, paths["delta"])
    return {
        "comparison": "B2024 minus A2024 only",
        "comparisons": comparisons,
        "diagnostic_GO": diagnostic_go,
        "recommendation": "B_and_A" if diagnostic_go else "A_over_B",
        "G2_delta_nonzero_rows": int(np.count_nonzero(delta[GROUP].to_numpy())),
        "G1_G3_delta_zero_and_A_B_bit_exact": True,
        "artifacts": {name: describe_file(path) for name, path in paths.items()},
        "diagnostic_did_not_change_candidates": True,
    }


def _validate_csv(path: Path, sample: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    if not path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise AssertionError("CSV BOM differs")
    text = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    if tuple(text.columns) != tuple(sample.columns) or len(text) != 8760:
        raise AssertionError("CSV schema/rows differs")
    if not text[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]].astype("string")):
        raise AssertionError("CSV sample identity differs")
    for group in TARGETS:
        expected = prediction[group].reset_index(drop=True).map(lambda value: f"{float(value):.6f}").astype("string")
        if not text[group].equals(expected):
            raise AssertionError(f"CSV six-decimal roundtrip differs: {group}")
    values = text.loc[:, list(TARGETS)].astype(np.float64).to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError("CSV contains non-finite values")
    for position, group in enumerate(TARGETS):
        if values[:, position].min() < 0 or values[:, position].max() > 1.02 * CAPACITY[group] + 5e-7:
            raise AssertionError(f"CSV bounds differ: {group}")
    return {"rows": 8760, "schema": True, "sample_id_time": True, "BOM": True, "six_decimals": True, "finite": True, "bounds": True, "sha256": sha256_file(path)}


def _final(v1: Mapping[str, Any], v2: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    specs = v1["input_identities"]
    index = _year_index(2025)
    base = _read_prediction(_verify(specs["final_recent_v4"]), index)
    rescue = _read_prediction(_verify(specs["final_g2_rescue_v5"]), index)
    gates = pd.read_parquet(_verify(specs["final_g2_rescue_gates"]))["gate"].to_numpy(dtype=bool)
    if int(gates.sum()) != int(specs["final_g2_rescue_gates"]["expected_gate_rows"]):
        raise AssertionError("frozen gate count differs")
    A, B, delta = candidates(base, rescue)
    changed = delta[GROUP].to_numpy() != 0.0
    if not np.array_equal(changed, gates):
        raise AssertionError("frozen delta/gates differ")
    if int(changed.sum()) != 143:
        raise AssertionError("final B affected-row count differs")
    for group in TARGETS:
        bound = 1.02 * CAPACITY[group]
        expected_A = np.clip(FACTOR * base[group].to_numpy(), 0.0, bound)
        expected_B = np.clip(
            expected_A + (rescue[group].to_numpy() - base[group].to_numpy()),
            0.0, bound,
        )
        if not np.array_equal(np.ascontiguousarray(expected_A).view(np.uint64), _bits(A[group])):
            raise AssertionError(f"A formula differs: {group}")
        if not np.array_equal(np.ascontiguousarray(expected_B).view(np.uint64), _bits(B[group])):
            raise AssertionError(f"B formula differs: {group}")
    A_path = out_dir / v2["output_contract"]["A_prediction"]
    B_path = out_dir / v2["output_contract"]["B_prediction"]
    delta_path = out_dir / "final/frozen_g2_rescue_delta.parquet"
    _atomic_parquet(A, A_path)
    _atomic_parquet(B, B_path)
    _atomic_parquet(delta, delta_path)
    for original, path in ((A, A_path), (B, B_path)):
        saved = pd.read_parquet(path).astype(np.float64)
        for group in TARGETS:
            if not np.array_equal(_bits(original[group]), _bits(saved[group])):
                raise AssertionError(f"parquet roundtrip differs: {path}/{group}")
    sample = pd.read_csv(
        _verify(specs["sample"]), encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGETS) or len(sample) != 8760:
        raise AssertionError("sample schema/rows differs")
    if not pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"]), name="forecast_kst_dtm").equals(index):
        raise AssertionError("sample time differs")
    outputs: dict[str, Any] = {}
    for name, prediction, csv_name in (
        ("A", A, v2["output_contract"]["A_csv"]),
        ("B", B, v2["output_contract"]["B_csv"]),
    ):
        submission = sample.copy()
        for group in TARGETS:
            submission[group] = prediction[group].to_numpy(dtype=np.float64)
        csv_path = out_dir / csv_name
        _atomic_csv(submission, csv_path)
        outputs[name] = {
            "prediction": describe_file(A_path if name == "A" else B_path),
            "submission": describe_file(csv_path),
            "validation": _validate_csv(csv_path, sample, prediction),
        }
    if sha256_file(out_dir / v2["output_contract"]["A_csv"]) == sha256_file(out_dir / v2["output_contract"]["B_csv"]):
        raise AssertionError("A/B CSVs unexpectedly duplicate")
    return {
        "A_formula_exact": True, "B_formula_exact": True,
        "frozen_G2_gate_rows": 143, "A_B_G1_G3_float64_bits_exact": True,
        "A_B_G2_difference_exactly_frozen_gate": True,
        "delta": describe_file(delta_path), "outputs": outputs,
    }


def _manifest(args: argparse.Namespace, v1: Mapping[str, Any], v2: Mapping[str, Any], diagnostic: Mapping[str, Any], final: Mapping[str, Any], closure: Mapping[str, Any]) -> None:
    _assert_closure(closure, _closure(args))
    files = sorted(
        (path for path in args.out_dir.rglob("*") if path.is_file() and path.name != "manifest.json"),
        key=lambda path: path.relative_to(args.out_dir).as_posix(),
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "public_adaptive_scale097_plain_and_frozen_g2_delta_v2",
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA,
        "v1_config_sha256": V1_SHA,
        "risk": v2["risk_classification"],
        "source_provenance": {"recursive_AST_closure": closure, "locked_before_metric_and_materialization": True, "unchanged_before_manifest": True},
        "public_feedback": v1["public_feedback_used"],
        "deduplication_and_feasibility": v1["deduplication_and_feasibility"],
        "immutable_candidates": v2["immutable_candidates_repeated"],
        "interaction_diagnostic": diagnostic,
        "final": final,
        "strict_or_private_claim": False,
        "outputs": [describe_file(path) for path in files],
        "output_count_excluding_manifest": len(files),
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
    }
    _write_json(args.out_dir / "manifest.json", manifest)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    v2, v1 = _verify_configs(args)
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    _copy(args.config, args.out_dir / "preregister_v2.json")
    _copy(args.config.with_suffix(".sha256"), args.out_dir / "preregister_v2.sha256")
    _copy(args.v1_config, args.out_dir / "preregister_v1.json")
    _copy(args.v1_config.with_suffix(".sha256"), args.out_dir / "preregister_v1.sha256")
    closure = _closure(args)
    closure_path = args.out_dir / "source_closure_before_metric_and_materialization_lock.json"
    _write_json(
        closure_path,
        {"schema_version": 1, "created_utc": utc_now(), "config_sha256": CONFIG_SHA, "A_B_values_materialized_before_lock": 0, "A_B_metric_values_computed_before_lock": 0, "A_B_CSV_bytes_before_lock": 0, "closure": closure},
    )
    diagnostic = _diagnostic(v1, args.out_dir)
    diagnostic_path = args.out_dir / "interaction_diagnostic_2024.json"
    _write_json(diagnostic_path, diagnostic)
    final = _final(v1, v2, args.out_dir)
    result_path = args.out_dir / "final_results.json"
    _write_json(
        result_path,
        {"schema_version": 1, "created_utc": utc_now(), "public_adaptive": True, "selection_unsafe": True, "strict_final_isolation": False, "private_champion": False, "leaderboard_score_claim": False, "diagnostic_recommendation": diagnostic["recommendation"], **final},
    )
    _manifest(args, v1, v2, diagnostic, final, closure)
    print(f"diagnostic_GO={diagnostic['diagnostic_GO']} recommendation={diagnostic['recommendation']}", flush=True)
    print(f"A_csv_sha256={final['outputs']['A']['submission']['sha256']}", flush=True)
    print(f"B_csv_sha256={final['outputs']['B']['submission']['sha256']}", flush=True)
    print(f"manifest_sha256={sha256_file(args.out_dir / 'manifest.json')}", flush=True)


if __name__ == "__main__":
    main()

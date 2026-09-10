"""Independent read-only audit of the frozen 0.97 plain/G2-delta pair."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.manifest import sha256_file, utc_now  # noqa: E402


ROOT = PROJECT_DIR / "artifacts/postgate/public_adaptive_scale097_g2_delta_v2"
V1_PATH = PROJECT_DIR / "configs/public_adaptive_scale097_g2_delta_preregister_v1.json"
V2_PATH = PROJECT_DIR / "configs/public_adaptive_scale097_g2_delta_preregister_v2.json"
V1_SHA = "5c637901229dbe3cde68a2c48123e3cf41acb86f88ec7c3db44e3f8322de459b"
V2_SHA = "9ed9bc8d4ec73f714cbd4d1b1c41ac8b5c45a7812a1bd2bbe28a8c5b5b24c475"
FROZEN_MANIFEST_SHA = "3ef6fabf059f3bd67e3b933e8eb633e5ebf3b422d7c123f1b27af461fcc2e153"
TARGETS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
GROUP = "kpx_group_2"
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _verify(spec: dict[str, Any]) -> Path:
    path = _path(str(spec["path"]))
    size = spec.get("bytes", spec.get("size_bytes"))
    if size is not None and path.stat().st_size != int(size):
        raise AssertionError(f"size mismatch: {path}")
    if sha256_file(path) != spec["sha256"]:
        raise AssertionError(f"hash mismatch: {path}")
    return path


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


def independent_ast_closure(entry: Path) -> tuple[Path, ...]:
    queue = [entry.resolve()]
    seen: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
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


def _bits(series: pd.Series) -> np.ndarray:
    return np.ascontiguousarray(series.to_numpy(dtype=np.float64)).view(np.uint64)


def _index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01 01:00", f"{year + 1}-01-01 00:00", freq="h", name="forecast_kst_dtm")


def _segments(year: int) -> dict[str, pd.DatetimeIndex]:
    def interval(start: str, end: str) -> pd.DatetimeIndex:
        return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")
    return {
        "full": _index(year),
        "H1": interval(f"{year}-01-01 01:00", f"{year}-07-01 00:00"),
        "H2": interval(f"{year}-07-01 01:00", f"{year + 1}-01-01 00:00"),
        "Q1": interval(f"{year}-01-01 01:00", f"{year}-04-01 00:00"),
        "Q2": interval(f"{year}-04-01 01:00", f"{year}-07-01 00:00"),
        "Q3": interval(f"{year}-07-01 01:00", f"{year}-10-01 00:00"),
        "Q4": interval(f"{year}-10-01 01:00", f"{year + 1}-01-01 00:00"),
    }


def _prediction(spec: dict[str, Any], year: int) -> pd.DataFrame:
    frame = pd.read_parquet(_verify(spec)).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGETS or not frame.index.equals(_index(year)):
        raise AssertionError("prediction schema/index mismatch")
    return frame


def _construct(base: pd.DataFrame, rescue: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    delta = rescue - base
    A = base.copy()
    B = base.copy()
    for group in TARGETS:
        a = np.clip(np.float64(0.97) * base[group].to_numpy(), 0, 1.02 * CAPACITY[group])
        b = np.clip(a + delta[group].to_numpy(), 0, 1.02 * CAPACITY[group])
        A[group], B[group] = a, b
    return A, B, delta


def _group(actual: pd.Series, prediction: pd.Series, group: str) -> tuple[float, float, float]:
    capacity = CAPACITY[group]
    y = actual.to_numpy(dtype=np.float64)
    p = prediction.to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & (y >= 0.10 * capacity)
    y, p = y[valid], p[valid]
    error = np.abs(p - y) / capacity
    n = 1.0 - float(error.mean())
    price = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    f = float(np.sum(y * price) / np.sum(y * 4.0))
    return 0.5 * (n + f), n, f


def _mixed(actual: pd.DataFrame, prediction: pd.DataFrame) -> tuple[float, float, float]:
    rows = [_group(actual[group], prediction[group], group) for group in TARGETS]
    n = float(np.mean([row[1] for row in rows]))
    f = float(np.mean([row[2] for row in rows]))
    return 0.5 * (n + f), n, f


def _assert_parquet(actual: pd.DataFrame, path: Path) -> None:
    saved = pd.read_parquet(path).astype(np.float64)
    if not saved.index.equals(actual.index) or tuple(saved.columns) != TARGETS:
        raise AssertionError(f"parquet schema mismatch: {path}")
    for group in TARGETS:
        if not np.array_equal(_bits(saved[group]), _bits(actual[group])):
            raise AssertionError(f"parquet values mismatch: {path}/{group}")


def _assert_csv(path: Path, sample: pd.DataFrame, prediction: pd.DataFrame) -> None:
    if not path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise AssertionError("CSV BOM mismatch")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    if tuple(frame.columns) != tuple(sample.columns) or len(frame) != 8760:
        raise AssertionError("CSV schema/rows mismatch")
    if not frame[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]]):
        raise AssertionError("CSV sample identity mismatch")
    for group in TARGETS:
        expected = prediction[group].reset_index(drop=True).map(lambda value: f"{float(value):.6f}").astype("string")
        if not frame[group].equals(expected):
            raise AssertionError(f"CSV text mismatch: {group}")
    numeric = frame.loc[:, list(TARGETS)].astype(np.float64).to_numpy()
    if not np.isfinite(numeric).all():
        raise AssertionError("CSV finite mismatch")
    for pos, group in enumerate(TARGETS):
        if numeric[:, pos].min() < 0 or numeric[:, pos].max() > 1.02 * CAPACITY[group] + 5e-7:
            raise AssertionError("CSV bounds mismatch")


def main() -> None:
    before = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in ROOT.rglob("*") if path.is_file()}
    if sha256_file(V1_PATH) != V1_SHA or sha256_file(V2_PATH) != V2_SHA:
        raise AssertionError("config hash mismatch")
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    v2 = json.loads(V2_PATH.read_text(encoding="utf-8"))
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != V2_SHA or manifest["v1_config_sha256"] != V1_SHA:
        raise AssertionError("manifest config binding mismatch")
    if not manifest["risk"]["public_adaptive"] or not manifest["risk"]["selection_unsafe"] or manifest["risk"]["private_champion"]:
        raise AssertionError("risk disclosure mismatch")
    declared = set()
    for spec in manifest["outputs"]:
        path = _verify(spec)
        if ROOT not in path.parents:
            raise AssertionError("output escapes canonical")
        declared.add(path.relative_to(ROOT).as_posix())
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "manifest_frozen_postrun_v2.json"}
    }
    if declared != actual:
        raise AssertionError("manifest output closure mismatch")

    runner = PROJECT_DIR / v2["source_provenance_protocol"]["runner"]
    independent = independent_ast_closure(runner)
    relative = [path.relative_to(PROJECT_DIR).as_posix() for path in independent]
    closure = manifest["source_provenance"]["recursive_AST_closure"]
    if relative != closure["resolved_relative_paths"]:
        raise AssertionError("independent AST closure mismatch")
    source_specs = {Path(spec["path"]).resolve().relative_to(PROJECT_DIR).as_posix(): spec for spec in closure["resolved_files"]}
    if set(source_specs) != set(relative):
        raise AssertionError("source mapping mismatch")
    for path in independent:
        _verify(source_specs[path.relative_to(PROJECT_DIR).as_posix()])
    locked = json.loads((ROOT / "source_closure_before_metric_and_materialization_lock.json").read_text(encoding="utf-8"))
    if locked["closure"] != closure or any(locked[key] != 0 for key in ("A_B_values_materialized_before_lock", "A_B_metric_values_computed_before_lock", "A_B_CSV_bytes_before_lock")):
        raise AssertionError("pre-materialization closure lock mismatch")

    frozen_manifest_path = ROOT / "manifest_frozen_postrun_v2.json"
    if sha256_file(frozen_manifest_path) != FROZEN_MANIFEST_SHA:
        raise AssertionError("frozen postrun manifest hash mismatch")
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    if frozen_manifest["authoritative_manifest"] is not True or frozen_manifest["candidate_and_CSV_recomputed"] is not False:
        raise AssertionError("frozen postrun manifest contract mismatch")
    current_closure = frozen_manifest["current_producer_source_config_test_closure"]
    if current_closure["resolved_relative_paths"] != relative:
        raise AssertionError("current postrun runtime path set mismatch")
    for spec in current_closure["resolved_files"]:
        _verify(spec)
    _verify(current_closure["test"])
    frozen_declared = {
        _verify(spec).relative_to(ROOT).as_posix()
        for spec in frozen_manifest["outputs_and_historical_records"]
    }
    frozen_actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "manifest_frozen_postrun_v2.json"
    }
    if frozen_declared != frozen_actual:
        raise AssertionError("frozen postrun output closure mismatch")

    specs = v1["input_identities"]
    base24 = _prediction(specs["interaction_recent_v4_2024"], 2024)
    rescue24 = _prediction(specs["interaction_g2_rescue_2024"], 2024)
    A24, B24, delta24 = _construct(base24, rescue24)
    _assert_parquet(A24, ROOT / "diagnostic_2024/A_scale097.parquet")
    _assert_parquet(B24, ROOT / "diagnostic_2024/B_scale097_plus_g2_delta.parquet")
    _assert_parquet(delta24, ROOT / "diagnostic_2024/frozen_rescue_delta.parquet")
    labels = pd.read_csv(_verify(specs["labels"]))
    labels.index = pd.DatetimeIndex(pd.to_datetime(labels.pop("kst_dtm")), name="forecast_kst_dtm")
    labels = labels.astype(np.float64)
    diagnostic = json.loads((ROOT / "interaction_diagnostic_2024.json").read_text(encoding="utf-8"))
    go = True
    exact_deltas: dict[str, Any] = {}
    for name, rows in _segments(2024).items():
        ga, _, _ = _group(labels.loc[rows, GROUP], A24.loc[rows, GROUP], GROUP)
        gb, _, _ = _group(labels.loc[rows, GROUP], B24.loc[rows, GROUP], GROUP)
        ma = _mixed(labels.loc[rows], A24.loc[rows])
        mb = _mixed(labels.loc[rows], B24.loc[rows])
        gd, md, nd, fd = gb - ga, mb[0] - ma[0], mb[1] - ma[1], mb[2] - ma[2]
        recorded = diagnostic["comparisons"][name]
        for actual_value, recorded_key in ((gd, "G2_delta_total_score"), (md, "mixed_delta_total_score"), (nd, "mixed_delta_one_minus_nmae"), (fd, "mixed_delta_ficr")):
            if not np.isclose(actual_value, recorded[recorded_key], atol=2e-15, rtol=0):
                raise AssertionError(f"diagnostic metric mismatch: {name}/{recorded_key}")
        go = go and gd > 0 and md > 0
        exact_deltas[name] = {"G2": gd, "mixed": md, "NMAE": nd, "FICR": fd}
    go = bool(go and exact_deltas["full"]["NMAE"] >= 0 and exact_deltas["full"]["FICR"] >= 0)
    if go or diagnostic["diagnostic_GO"] or diagnostic["recommendation"] != "A_over_B":
        raise AssertionError("diagnostic NO-GO/recommendation mismatch")

    base25 = _prediction(specs["final_recent_v4"], 2025)
    rescue25 = _prediction(specs["final_g2_rescue_v5"], 2025)
    A25, B25, delta25 = _construct(base25, rescue25)
    A_path = ROOT / v2["output_contract"]["A_prediction"]
    B_path = ROOT / v2["output_contract"]["B_prediction"]
    _assert_parquet(A25, A_path)
    _assert_parquet(B25, B_path)
    _assert_parquet(delta25, ROOT / "final/frozen_g2_rescue_delta.parquet")
    gates = pd.read_parquet(_verify(specs["final_g2_rescue_gates"]))["gate"].to_numpy(dtype=bool)
    changed = delta25[GROUP].to_numpy() != 0.0
    if not np.array_equal(changed, gates) or int(changed.sum()) != 143:
        raise AssertionError("frozen 143-row delta mismatch")
    for group in ("kpx_group_1", "kpx_group_3"):
        if not np.array_equal(_bits(A25[group]), _bits(B25[group])):
            raise AssertionError("A/B identity group mismatch")
    sample = pd.read_csv(_verify(specs["sample"]), encoding="utf-8-sig", dtype="string")
    A_csv = ROOT / v2["output_contract"]["A_csv"]
    B_csv = ROOT / v2["output_contract"]["B_csv"]
    _assert_csv(A_csv, sample, A25)
    _assert_csv(B_csv, sample, B25)
    if sha256_file(A_csv) == sha256_file(B_csv):
        raise AssertionError("A/B duplicate hash")

    after = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in ROOT.rglob("*") if path.is_file()}
    if before != after:
        raise AssertionError("audit mutated canonical")
    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "audit_type": "independent_read_only_public_adaptive_scale097_g2_delta_v2",
        "status": "PASS",
        "config_v1_sha256": V1_SHA,
        "config_v2_sha256": V2_SHA,
        "historical_manifest_sha256": sha256_file(manifest_path),
        "authoritative_frozen_manifest_sha256": FROZEN_MANIFEST_SHA,
        "A_csv_sha256": sha256_file(A_csv),
        "B_csv_sha256": sha256_file(B_csv),
        "checks": {
            "manifest_output_closure": True,
            "independent_recursive_AST_closure": relative,
            "pre_materialization_zero_lock": True,
            "postrun_test_current_and_7_pass_frozen_manifest_closure": True,
            "A_B_2024_formula_and_exact_metrics": exact_deltas,
            "diagnostic_GO": False,
            "recommendation": "A_over_B",
            "A_B_2025_formulas_and_143_frozen_rows": True,
            "parquet_bits_CSV_sample_time_BOM_six_decimals_finite_bounds": True,
            "A_B_not_duplicates": True,
            "canonical_nonmutation": True,
        },
        "risk": manifest["risk"],
        "canonical_inventory_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
    }
    output = PROJECT_DIR / "artifacts/audits/public_adaptive_scale097_g2_delta_v2_postrun_independent.json"
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS {output} sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()

"""Independent read-only audit for the frozen scale097 plus FICR G1 delta artifact."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
CANONICAL = PROJECT_DIR / "artifacts/postgate/public_adaptive_scale097_ficr_g1_delta_v1"
CONFIG = PROJECT_DIR / "configs/public_adaptive_scale097_ficr_g1_delta_preregister_v1.json"
CONFIG_SHA = "30899a8496184f32de228edf154eab7764eb8c52144940f38e12db7d0c29fcb2"
MANIFEST_SHA = "4792b4adf4c9a69b47856fcde9823c7358cb83b429c8fbfff91e4768e7866afc"
CSV_SHA = "1113fd5a5e27fdb3151fdf944e41f8a0a23e9bbb80d37e9d0837bc5e67ea0c8f"
TARGETS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/audits/public_adaptive_scale097_ficr_g1_delta_v1_local.json"),
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(spec: Mapping[str, Any]) -> Path:
    path = Path(str(spec["path"]))
    path = path if path.is_absolute() else PROJECT_DIR / path
    expected_size = spec.get("bytes", spec.get("size_bytes"))
    assert path.is_file()
    if expected_size is not None:
        assert path.stat().st_size == int(expected_size)
    assert sha256(path) == str(spec["sha256"])
    return path


def local_files(module: str) -> set[Path]:
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


def ast_closure(entry: Path) -> tuple[Path, ...]:
    pending = [entry.resolve()]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        assert PROJECT_DIR in path.parents
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    hits = local_files(alias.name)
                    if hits:
                        pending.extend(hits)
                    elif (PROJECT_DIR / alias.name.split(".")[0]).exists():
                        raise AssertionError(f"unresolved local import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                hits = local_files(base)
                if hits:
                    pending.extend(hits)
                elif base and (PROJECT_DIR / base.split(".")[0]).exists():
                    raise AssertionError(f"unresolved local from-import: {base}")
                for alias in node.names:
                    if alias.name != "*":
                        pending.extend(local_files(f"{base}.{alias.name}"))
    return tuple(sorted(seen))


def year_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{year}-01-01 01:00", f"{year + 1}-01-01 00:00", freq="h",
        name="forecast_kst_dtm",
    )


def segment_rows(year: int) -> dict[str, pd.DatetimeIndex]:
    def interval(start: str, end: str) -> pd.DatetimeIndex:
        return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")
    return {
        "full": year_index(year),
        "H1": interval(f"{year}-01-01 01:00", f"{year}-07-01 00:00"),
        "H2": interval(f"{year}-07-01 01:00", f"{year + 1}-01-01 00:00"),
        "Q1": interval(f"{year}-01-01 01:00", f"{year}-04-01 00:00"),
        "Q2": interval(f"{year}-04-01 01:00", f"{year}-07-01 00:00"),
        "Q3": interval(f"{year}-07-01 01:00", f"{year}-10-01 00:00"),
        "Q4": interval(f"{year}-10-01 01:00", f"{year + 1}-01-01 00:00"),
    }


def bits(series: pd.Series) -> np.ndarray:
    return np.ascontiguousarray(series.to_numpy(dtype=np.float64)).view(np.uint64)


def prediction(path: Path, year: int) -> pd.DataFrame:
    frame = pd.read_parquet(path).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    assert tuple(frame.columns) == TARGETS
    assert frame.index.equals(year_index(year))
    return frame


def labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm")), name="forecast_kst_dtm")
    assert tuple(frame.columns) == TARGETS
    return frame.astype(np.float64)


def group_metric(y: pd.Series, p: pd.Series, group: str) -> dict[str, float]:
    capacity = CAPACITY[group]
    actual = y.to_numpy(dtype=np.float64)
    estimate = p.to_numpy(dtype=np.float64)
    valid = np.isfinite(actual) & (actual >= 0.10 * capacity)
    actual, estimate = actual[valid], estimate[valid]
    error = np.abs(estimate - actual) / capacity
    n = 1.0 - float(error.mean())
    price = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    f = float(np.sum(actual * price) / np.sum(actual * 4.0))
    return {"total_score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f}


def mixed_metric(y: pd.DataFrame, p: pd.DataFrame) -> dict[str, float]:
    rows = [group_metric(y[group], p[group], group) for group in TARGETS]
    n = float(np.mean([row["one_minus_nmae"] for row in rows]))
    f = float(np.mean([row["ficr"] for row in rows]))
    return {"total_score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f}


def assert_close(left: float, right: float) -> None:
    assert np.isclose(left, right, rtol=0.0, atol=1e-15), (left, right)


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path if path.is_absolute() else PROJECT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    assert sha256(CONFIG) == CONFIG_SHA
    assert CONFIG.with_suffix(".sha256").read_text(encoding="utf-8") == f"{CONFIG_SHA}  {CONFIG.name}\n"
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest_path = CANONICAL / "manifest.json"
    assert sha256(manifest_path) == MANIFEST_SHA
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["risk"]["selection_unsafe"] is True
    assert manifest["risk"]["private_champion"] is False
    assert manifest["diagnostic"]["diagnostic_GO"] is False
    assert manifest["diagnostic"]["recommendation"] == "A_over_candidate"
    for spec in config["lineage"].values():
        verify(spec)
    for spec in config["input_identities"].values():
        verify(spec)
    for spec in manifest["outputs"]:
        verify(spec)

    independent_closure = ast_closure(PROJECT_DIR / "scripts/run_public_adaptive_scale097_ficr_g1_delta.py")
    closure = manifest["source_provenance"]["recursive_AST_closure"]
    relatives = [path.relative_to(PROJECT_DIR).as_posix() for path in independent_closure]
    assert relatives == closure["resolved_relative_paths"]
    assert len(independent_closure) == closure["resolved_file_count"]
    for path, spec in zip(independent_closure, closure["resolved_files"], strict=True):
        assert path == Path(spec["path"])
        verify(spec)
    verify(closure["test"])

    before = json.loads((CANONICAL / "protected_upstream_snapshot_before_metric_and_CSV.json").read_text(encoding="utf-8"))
    after = json.loads((CANONICAL / "protected_upstream_snapshot_after_metric_and_CSV.json").read_text(encoding="utf-8"))
    for key in ("protected_roots", "protected_files", "file_count", "files"):
        assert before[key] == after[key]
    assert after["exactly_equal_to_before"] is True
    for spec in before["files"]:
        path = PROJECT_DIR / spec["relative_path"]
        assert path.stat().st_size == int(spec["size_bytes"])
        assert sha256(path) == spec["sha256"]

    specs = config["input_identities"]
    A24 = prediction(verify(specs["A_2024_scale097"]), 2024)
    ref24 = prediction(verify(specs["ficr_delta_reference_2024"]), 2024)
    adj24 = prediction(verify(specs["ficr_delta_adjusted_2024"]), 2024)
    delta24 = adj24 - ref24
    for group in ("kpx_group_2", "kpx_group_3"):
        assert np.all(delta24[group].to_numpy() == 0.0)
    expected24 = A24.copy()
    expected24["kpx_group_1"] = np.clip(
        A24["kpx_group_1"].to_numpy() + delta24["kpx_group_1"].to_numpy(),
        0.0,
        1.02 * CAPACITY["kpx_group_1"],
    )
    saved24 = prediction(CANONICAL / config["output_contract"]["diagnostic_candidate"], 2024)
    for group in TARGETS:
        assert np.array_equal(bits(expected24[group]), bits(saved24[group]))
    actual = labels(verify(specs["labels"]))
    stored = json.loads((CANONICAL / "interaction_diagnostic_2024.json").read_text(encoding="utf-8"))
    all_positive = True
    recomputed_deltas: dict[str, float] = {}
    for name, rows in segment_rows(2024).items():
        ga = group_metric(actual.loc[rows, "kpx_group_1"], A24.loc[rows, "kpx_group_1"], "kpx_group_1")
        gb = group_metric(actual.loc[rows, "kpx_group_1"], expected24.loc[rows, "kpx_group_1"], "kpx_group_1")
        ma = mixed_metric(actual.loc[rows], A24.loc[rows])
        mb = mixed_metric(actual.loc[rows], expected24.loc[rows])
        gd = gb["total_score"] - ga["total_score"]
        md = mb["total_score"] - ma["total_score"]
        record = stored["comparisons"][name]
        for key, value in ga.items():
            assert_close(value, record["G1_A"][key])
        for key, value in gb.items():
            assert_close(value, record["G1_candidate"][key])
        for key, value in ma.items():
            assert_close(value, record["mixed_A"][key])
        for key, value in mb.items():
            assert_close(value, record["mixed_candidate"][key])
        assert_close(gd, record["G1_delta_total_score"])
        assert_close(md, record["mixed_delta_total_score"])
        recomputed_deltas[name] = md
        all_positive = all_positive and gd > 0.0 and md > 0.0
    full_n = stored["comparisons"]["full"]["mixed_delta_one_minus_nmae"]
    full_f = stored["comparisons"]["full"]["mixed_delta_ficr"]
    assert bool(all_positive and full_n >= 0.0 and full_f >= 0.0) is False

    A25 = prediction(verify(specs["A_final_scale097"]), 2025)
    reference_csv = pd.read_csv(verify(specs["ficr_delta_reference_final_csv"]), encoding="utf-8-sig")
    ref25 = reference_csv.loc[:, list(TARGETS)].astype(np.float64)
    ref25.index = year_index(2025)
    adj25 = prediction(verify(specs["ficr_composed_final_direct"]), 2025)
    delta25 = adj25 - ref25
    for group in ("kpx_group_2", "kpx_group_3"):
        assert np.all(delta25[group].to_numpy() == 0.0)
    expected25 = A25.copy()
    expected25["kpx_group_1"] = np.clip(
        A25["kpx_group_1"].to_numpy() + delta25["kpx_group_1"].to_numpy(),
        0.0,
        1.02 * CAPACITY["kpx_group_1"],
    )
    saved25 = prediction(CANONICAL / config["output_contract"]["final_prediction"], 2025)
    for group in TARGETS:
        assert np.array_equal(bits(expected25[group]), bits(saved25[group]))
    for group in ("kpx_group_2", "kpx_group_3"):
        assert np.array_equal(bits(A25[group]), bits(saved25[group]))

    csv_path = CANONICAL / config["output_contract"]["CSV"]
    assert sha256(csv_path) == CSV_SHA
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    sample = pd.read_csv(verify(specs["sample"]), encoding="utf-8-sig", dtype="string")
    rendered = pd.read_csv(csv_path, encoding="utf-8-sig", dtype="string")
    assert len(rendered) == 8760
    assert tuple(rendered.columns) == tuple(sample.columns)
    assert rendered[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]])
    for group in TARGETS:
        expected_text = expected25[group].reset_index(drop=True).map(lambda value: f"{float(value):.6f}").astype("string")
        assert rendered[group].equals(expected_text)
    numeric = rendered.loc[:, list(TARGETS)].astype(np.float64).to_numpy()
    assert np.isfinite(numeric).all()
    for position, group in enumerate(TARGETS):
        assert numeric[:, position].min() >= 0.0
        assert numeric[:, position].max() <= 1.02 * CAPACITY[group] + 5e-7

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "read_only": True,
        "config_sha256": CONFIG_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "CSV_sha256": CSV_SHA,
        "risk": {"public_adaptive": True, "selection_unsafe": True, "private_champion": False},
        "source_AST_closure_exact": True,
        "test_bound": True,
        "protected_upstream_before_after_and_current_exact": True,
        "formula_2024_exact": True,
        "formula_2025_exact": True,
        "G2_G3_bit_identity": True,
        "all_seven_metrics_recomputed_exact": True,
        "diagnostic_GO": False,
        "recommendation": "A_over_candidate",
        "mixed_total_score_delta": recomputed_deltas,
        "CSV_validation": {
            "rows": 8760,
            "sample_schema_id_time": True,
            "BOM": True,
            "six_decimal_roundtrip": True,
            "finite_bounds": True,
        },
        "canonical_nonmutation": True,
    }
    write_atomic(args.out, payload)
    output = args.out if args.out.is_absolute() else PROJECT_DIR / args.out
    print(f"status=PASS audit_sha256={sha256(output)}")


if __name__ == "__main__":
    main()

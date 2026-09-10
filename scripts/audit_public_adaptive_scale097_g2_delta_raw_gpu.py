"""Independent read-only audit for the frozen scale-0.97 A/B pair."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/postgate/public_adaptive_scale097_g2_delta_v2"
FROZEN_MANIFEST_SHA = "3ef6fabf059f3bd67e3b933e8eb633e5ebf3b422d7c123f1b27af461fcc2e153"
ORIGINAL_MANIFEST_SHA = "8e05b6edf775d33f564c5ae5c656055aef51200c52ae78294a28392ac263c821"
V2_SHA = "9ed9bc8d4ec73f714cbd4d1b1c41ac8b5c45a7812a1bd2bbe28a8c5b5b24c475"
A_CSV_SHA = "e8f848b9a08c44b2552367f8e83ae4c9e132c8cf416ea8c437fd7a7e26314f58"
B_CSV_SHA = "9960eb56b8aea708f8fa2cc4599526ba44fc3259a153a2dec9ea2a61222260bf"
GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_exact(record: Mapping[str, Any]) -> None:
    path = Path(str(record["path"]))
    size = int(record.get("size_bytes", record.get("bytes")))
    if not path.is_file() or path.stat().st_size != size or sha256(path) != record["sha256"]:
        raise AssertionError(f"file record differs: {path}")


def assert_close(actual: Any, expected: Any, context: str) -> None:
    if isinstance(expected, Mapping):
        if set(actual) != set(expected):
            raise AssertionError(f"{context}: keys differ")
        for key in expected:
            assert_close(actual[key], expected[key], f"{context}/{key}")
    elif isinstance(expected, (int, np.integer)) and not isinstance(expected, bool):
        if int(actual) != int(expected):
            raise AssertionError(f"{context}: integer differs")
    elif isinstance(expected, (float, np.floating)):
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=2e-14):
            raise AssertionError(f"{context}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(f"{context}: {actual!r} != {expected!r}")


def module_files(module: str) -> set[Path]:
    parts = module.split(".") if module else []
    found: set[Path] = set()
    if parts:
        file_path = ROOT.joinpath(*parts).with_suffix(".py")
        init_path = ROOT.joinpath(*parts, "__init__.py")
        if file_path.is_file():
            found.add(file_path.resolve())
        if init_path.is_file():
            found.add(init_path.resolve())
        for depth in range(1, len(parts)):
            init = ROOT.joinpath(*parts[:depth], "__init__.py")
            if init.is_file():
                found.add(init.resolve())
    return found


def ast_closure(entrypoint: Path) -> tuple[set[Path], list[str]]:
    queue = [entrypoint.resolve()]
    visited: set[Path] = set()
    dynamic: list[str] = []
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "__import__"}:
                    dynamic.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.func.id}")
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "importlib"
                    and node.func.attr == "import_module"
                ):
                    dynamic.append(f"{path.relative_to(ROOT)}:{node.lineno}:import_module")
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.append(node.module)
                package = ROOT.joinpath(*node.module.split("."))
                if package.is_dir():
                    modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
            for module in modules:
                queue.extend(module_files(module).difference(visited))
    return visited, dynamic


def year_segments(year: int) -> dict[str, pd.DatetimeIndex]:
    make = lambda a, b: pd.date_range(a, b, freq="h", name="forecast_kst_dtm")
    return {
        "full": make(f"{year}-01-01 01:00", f"{year + 1}-01-01 00:00"),
        "H1": make(f"{year}-01-01 01:00", f"{year}-07-01 00:00"),
        "H2": make(f"{year}-07-01 01:00", f"{year + 1}-01-01 00:00"),
        "Q1": make(f"{year}-01-01 01:00", f"{year}-04-01 00:00"),
        "Q2": make(f"{year}-04-01 01:00", f"{year}-07-01 00:00"),
        "Q3": make(f"{year}-07-01 01:00", f"{year}-10-01 00:00"),
        "Q4": make(f"{year}-10-01 01:00", f"{year + 1}-01-01 00:00"),
    }


def group_metrics(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, float]:
    capacity = CAPACITY[group]
    y = actual.to_numpy(np.float64)
    p = prediction.to_numpy(np.float64)
    valid = np.isfinite(y) & (y >= 0.10 * capacity)
    y, p = y[valid], p[valid]
    error = np.abs(p - y) / capacity
    one_minus = 1.0 - float(np.mean(error))
    unit = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    ficr = float(np.sum(y * unit) / np.sum(y * 4.0))
    return {"total_score": 0.5 * (one_minus + ficr), "one_minus_nmae": one_minus, "ficr": ficr}


def mixed_metrics(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, float]:
    rows = [group_metrics(actual[g], prediction[g], g) for g in GROUPS]
    one_minus = float(np.mean([row["one_minus_nmae"] for row in rows]))
    ficr = float(np.mean([row["ficr"] for row in rows]))
    return {"total_score": 0.5 * (one_minus + ficr), "one_minus_nmae": one_minus, "ficr": ficr}


def read_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    return frame


def bits(series: pd.Series) -> np.ndarray:
    return np.ascontiguousarray(series.to_numpy(np.float64)).view(np.uint64)


def main() -> None:
    frozen_path = ARTIFACT / "manifest_frozen_postrun_v2.json"
    if sha256(frozen_path) != FROZEN_MANIFEST_SHA:
        raise AssertionError("frozen manifest hash differs")
    if sha256(ARTIFACT / "manifest.json") != ORIGINAL_MANIFEST_SHA:
        raise AssertionError("historical manifest hash differs")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    historical = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
    before = {
        "frozen": sha256(frozen_path),
        "historical": sha256(ARTIFACT / "manifest.json"),
        "v5": sha256(ROOT / "artifacts/postgate/direct_interval_g2_posthoc_rescue_v5/manifest.json"),
    }

    listed: set[Path] = set()
    for record in frozen["outputs_and_historical_records"]:
        file_exact(record)
        listed.add(Path(record["path"]).resolve())
    actual = {path.resolve() for path in ARTIFACT.rglob("*") if path.is_file() and path != frozen_path}
    if listed != actual or len(actual) != frozen["output_count_excluding_this_manifest"]:
        raise AssertionError("frozen manifest output closure differs")
    for record in (
        frozen["postrun_test_addendum"],
        frozen["postrun_test_addendum_sidecar"],
        frozen["supersedes_historical_manifest_for_current_test_provenance"],
    ):
        file_exact(record)

    current = frozen["current_producer_source_config_test_closure"]
    independently_resolved, dynamic = ast_closure(ROOT / current["entrypoint"])
    relative = sorted(path.relative_to(ROOT).as_posix() for path in independently_resolved)
    if dynamic or relative != current["resolved_relative_paths"]:
        raise AssertionError("independent AST closure differs")
    locked_sources = {Path(row["path"]).resolve(): row for row in current["resolved_files"]}
    if set(locked_sources) != independently_resolved or len(independently_resolved) != 3:
        raise AssertionError("AST source set differs")
    for record in [*locked_sources.values(), current["test"], current["v1_config"], current["v1_sidecar"], current["v2_config"], current["v2_sidecar"]]:
        file_exact(record)
    if current["v2_config"]["sha256"] != V2_SHA or current["unresolved_local_imports"]:
        raise AssertionError("v2 config/unresolved imports differ")
    if frozen["focused_postrun_tests"] != {"command": ".venv/Scripts/python.exe -m pytest -q tests/test_public_adaptive_scale097_g2_delta.py", "failed": 0, "passed": 7}:
        raise AssertionError("focused postrun test record differs")

    # Verify all registered v1 input identities independently.
    v1 = json.loads((ROOT / "configs/public_adaptive_scale097_g2_delta_preregister_v1.json").read_text(encoding="utf-8"))
    lineage = [
        record
        for name, record in v1["lineage"].items()
        if name != "g2_v5_local_independent_audit"
    ]
    lineage.append(
        json.loads(
            (ROOT / "configs/public_adaptive_scale097_g2_delta_preregister_v2.json").read_text(
                encoding="utf-8"
            )
        )["current_v5_independent_audit"]
    )
    for record in [*lineage, *v1["input_identities"].values()]:
        file_exact(record)

    capacities = np.asarray([CAPACITY[g] for g in GROUPS], dtype=np.float64)
    final_recent = read_prediction(ROOT / v1["input_identities"]["final_recent_v4"]["path"])
    final_rescue = read_prediction(ROOT / v1["input_identities"]["final_g2_rescue_v5"]["path"])
    gates = pd.read_parquet(ROOT / v1["input_identities"]["final_g2_rescue_gates"]["path"])["gate"].to_numpy(bool)
    stored_delta = read_prediction(ARTIFACT / "final/frozen_g2_rescue_delta.parquet")
    A = read_prediction(ARTIFACT / "final/A_plain_scale097.parquet")
    B = read_prediction(ARTIFACT / "final/B_scale097_plus_g2_rescue_delta.parquet")
    expected_delta = final_rescue - final_recent
    expected_A = pd.DataFrame(
        np.clip(np.float64(0.97) * final_recent.to_numpy(np.float64), 0.0, 1.02 * capacities),
        index=final_recent.index,
        columns=GROUPS,
    )
    expected_B = pd.DataFrame(
        np.clip(expected_A.to_numpy(np.float64) + expected_delta.to_numpy(np.float64), 0.0, 1.02 * capacities),
        index=final_recent.index,
        columns=GROUPS,
    )
    for group in GROUPS:
        if not np.array_equal(bits(stored_delta[group]), bits(expected_delta[group])):
            raise AssertionError(f"final delta differs: {group}")
        if not np.array_equal(bits(A[group]), bits(expected_A[group])):
            raise AssertionError(f"A formula differs: {group}")
        if not np.array_equal(bits(B[group]), bits(expected_B[group])):
            raise AssertionError(f"B formula differs: {group}")
    if np.count_nonzero(stored_delta["kpx_group_1"].to_numpy()) or np.count_nonzero(stored_delta["kpx_group_3"].to_numpy()):
        raise AssertionError("final G1/G3 delta is not exact zero")
    nonzero = stored_delta["kpx_group_2"].to_numpy() != 0.0
    if int(nonzero.sum()) != 143 or not np.array_equal(nonzero, gates) or int(gates.sum()) != 143:
        raise AssertionError("final frozen G2 143-row gate differs")

    # Rebuild the complete 2024 diagnostic and all seven metrics.
    base24 = read_prediction(ROOT / v1["input_identities"]["interaction_recent_v4_2024"]["path"])
    rescue24 = read_prediction(ROOT / v1["input_identities"]["interaction_g2_rescue_2024"]["path"])
    delta24 = rescue24 - base24
    A24_expected = pd.DataFrame(
        np.clip(np.float64(0.97) * base24.to_numpy(np.float64), 0.0, 1.02 * capacities),
        index=base24.index,
        columns=GROUPS,
    )
    B24_expected = pd.DataFrame(
        np.clip(A24_expected.to_numpy(np.float64) + delta24.to_numpy(np.float64), 0.0, 1.02 * capacities),
        index=base24.index,
        columns=GROUPS,
    )
    A24 = read_prediction(ARTIFACT / "diagnostic_2024/A_scale097.parquet")
    B24 = read_prediction(ARTIFACT / "diagnostic_2024/B_scale097_plus_g2_delta.parquet")
    D24 = read_prediction(ARTIFACT / "diagnostic_2024/frozen_rescue_delta.parquet")
    for group in GROUPS:
        for observed, expected, name in ((A24, A24_expected, "A24"), (B24, B24_expected, "B24"), (D24, delta24, "D24")):
            if not np.array_equal(bits(observed[group]), bits(expected[group])):
                raise AssertionError(f"{name}/{group} differs")
    if np.count_nonzero(D24["kpx_group_1"].to_numpy()) or np.count_nonzero(D24["kpx_group_3"].to_numpy()) or np.count_nonzero(D24["kpx_group_2"].to_numpy()) != 199:
        raise AssertionError("2024 delta support differs")
    labels = pd.read_csv(
        r"data/local/open/train/train_labels.csv",
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    stored_metrics = json.loads((ARTIFACT / "interaction_diagnostic_2024.json").read_text(encoding="utf-8"))
    minimum_g2 = float("inf")
    minimum_mixed = float("inf")
    recomputed: dict[str, Any] = {}
    for segment, index in year_segments(2024).items():
        g2a = group_metrics(labels.loc[index, "kpx_group_2"], A24.loc[index, "kpx_group_2"], "kpx_group_2")
        g2b = group_metrics(labels.loc[index, "kpx_group_2"], B24.loc[index, "kpx_group_2"], "kpx_group_2")
        ma = mixed_metrics(labels.loc[index, list(GROUPS)], A24.loc[index])
        mb = mixed_metrics(labels.loc[index, list(GROUPS)], B24.loc[index])
        record = {
            "G2_A": g2a,
            "G2_B": g2b,
            "G2_delta_total_score": g2b["total_score"] - g2a["total_score"],
            "mixed_A": ma,
            "mixed_B": mb,
            "mixed_delta_total_score": mb["total_score"] - ma["total_score"],
            "mixed_delta_one_minus_nmae": mb["one_minus_nmae"] - ma["one_minus_nmae"],
            "mixed_delta_ficr": mb["ficr"] - ma["ficr"],
        }
        assert_close(stored_metrics["comparisons"][segment], record, f"metric/{segment}")
        minimum_g2 = min(minimum_g2, record["G2_delta_total_score"])
        minimum_mixed = min(minimum_mixed, record["mixed_delta_total_score"])
        recomputed[segment] = record
    full = recomputed["full"]
    go = all(row["G2_delta_total_score"] > 0.0 and row["mixed_delta_total_score"] > 0.0 for row in recomputed.values()) and full["mixed_delta_one_minus_nmae"] >= 0.0 and full["mixed_delta_ficr"] >= 0.0
    if go or stored_metrics["diagnostic_GO"] or stored_metrics["recommendation"] != "A_over_B":
        raise AssertionError("diagnostic gate/recommendation differs")
    if historical["interaction_diagnostic"]["comparisons"] != stored_metrics["comparisons"]:
        raise AssertionError("manifest diagnostic differs")

    sample = pd.read_csv(r"data/local/open/sample_submission.csv", encoding="utf-8-sig", dtype="string")
    csv_specs = {
        "A": (ARTIFACT / "corrected_recent_v4_scale_097_public_adaptive_2025.csv", A, A_CSV_SHA),
        "B": (ARTIFACT / "corrected_recent_v4_scale_097_plus_g2_rescue_delta_2025.csv", B, B_CSV_SHA),
    }
    for name, (path, prediction, expected_hash) in csv_specs.items():
        raw = path.read_bytes()
        if sha256(path) != expected_hash or raw[:3] != b"\xef\xbb\xbf":
            raise AssertionError(f"{name}: CSV hash/BOM differs")
        text = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
        if tuple(text.columns) != tuple(sample.columns) or len(text) != 8760:
            raise AssertionError(f"{name}: CSV schema/rows differ")
        if not text[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]]):
            raise AssertionError(f"{name}: CSV IDs/time differ")
        for group in GROUPS:
            expected = prediction[group].map(lambda value: f"{float(value):.6f}").astype("string")
            if not text[group].equals(expected.reset_index(drop=True)):
                raise AssertionError(f"{name}: CSV six-decimal roundtrip differs")
        numeric = text.loc[:, list(GROUPS)].astype(np.float64).to_numpy()
        if not np.isfinite(numeric).all() or np.any(numeric < 0.0) or np.any(numeric > 1.02 * capacities + 5e-7):
            raise AssertionError(f"{name}: CSV bounds/finite differ")

    after = {
        "frozen": sha256(frozen_path),
        "historical": sha256(ARTIFACT / "manifest.json"),
        "v5": sha256(ROOT / "artifacts/postgate/direct_interval_g2_posthoc_rescue_v5/manifest.json"),
    }
    if before != after:
        raise AssertionError("canonical changed during audit")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "artifact": "public_adaptive_scale097_g2_delta_v2",
        "authoritative_frozen_manifest_sha256": FROZEN_MANIFEST_SHA,
        "historical_manifest_sha256": ORIGINAL_MANIFEST_SHA,
        "preregister_v2_sha256": V2_SHA,
        "independent_AST": {"exact": True, "files": relative, "dynamic_imports": dynamic},
        "manifest_output_and_source_closure_exact": True,
        "current_test": {"sha256": current["test"]["sha256"], "focused_passed": 7, "focused_failed": 0},
        "final": {
            "A_formula_exact": True,
            "B_formula_exact": True,
            "delta_exact": True,
            "frozen_G2_rows": int(nonzero.sum()),
            "G1_G3_delta_zero_bits": True,
            "A_csv_sha256": A_CSV_SHA,
            "B_csv_sha256": B_CSV_SHA,
            "CSV_contracts_exact": True,
        },
        "diagnostic_2024": {
            "formula_and_delta_exact": True,
            "metric_records_recomputed": 7,
            "G2_delta_rows": int(np.count_nonzero(D24["kpx_group_2"].to_numpy())),
            "minimum_G2_delta": minimum_g2,
            "minimum_mixed_delta": minimum_mixed,
            "GO": go,
            "recommendation": "A_over_B",
        },
        "risk": frozen["risk"],
        "canonical_nonmutation": True,
    }
    destination = ROOT / "artifacts/audits/public_adaptive_scale097_g2_delta_v2_raw_gpu.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(f"PASS {destination} sha256={sha256(destination)}")


if __name__ == "__main__":
    main()

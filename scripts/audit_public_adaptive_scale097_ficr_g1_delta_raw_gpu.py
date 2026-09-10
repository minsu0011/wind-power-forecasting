"""Independent read-only audit of the frozen scale097 + direct FICR G1 delta."""

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
ARTIFACT = ROOT / "artifacts/postgate/public_adaptive_scale097_ficr_g1_delta_v1"
CONFIG = ROOT / "configs/public_adaptive_scale097_ficr_g1_delta_preregister_v1.json"
DESTINATION = ROOT / "artifacts/audits/public_adaptive_scale097_ficr_g1_delta_v1_raw_gpu.json"
CONFIG_SHA = "30899a8496184f32de228edf154eab7764eb8c52144940f38e12db7d0c29fcb2"
MANIFEST_SHA = "4792b4adf4c9a69b47856fcde9823c7358cb83b429c8fbfff91e4768e7866afc"
CSV_SHA = "1113fd5a5e27fdb3151fdf944e41f8a0a23e9bbb80d37e9d0837bc5e67ea0c8f"
PREVIOUS_AUDIT_SHA = "9e46b778af0131d576cf77a0ae3a94be3ca87e5549f2ecb0b472c6228def73e0"
GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def verify_record(record: Mapping[str, Any]) -> Path:
    path = resolve(str(record["path"]))
    expected_size = record.get("size_bytes", record.get("bytes"))
    if not path.is_file():
        raise AssertionError(f"missing file: {path}")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise AssertionError(f"size differs: {path}")
    if sha256(path) != str(record["sha256"]):
        raise AssertionError(f"SHA differs: {path}")
    return path


def module_files(module: str) -> set[Path]:
    parts = [part for part in module.split(".") if part]
    if not parts:
        return set()
    found: set[Path] = set()
    module_file = ROOT.joinpath(*parts).with_suffix(".py")
    package_init = ROOT.joinpath(*parts, "__init__.py")
    if module_file.is_file():
        found.add(module_file.resolve())
    if package_init.is_file():
        found.add(package_init.resolve())
    for depth in range(1, len(parts)):
        parent_init = ROOT.joinpath(*parts[:depth], "__init__.py")
        if parent_init.is_file():
            found.add(parent_init.resolve())
    return found


def ast_closure(entrypoint: Path) -> tuple[list[Path], list[str], list[str]]:
    pending = [entrypoint.resolve()]
    visited: set[Path] = set()
    dynamic: list[str] = []
    unresolved: list[str] = []
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "__import__"}:
                    dynamic.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}:{node.func.id}")
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "importlib"
                    and node.func.attr == "import_module"
                ):
                    dynamic.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}:import_module")
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.append(node.module)
                package = ROOT.joinpath(*node.module.split("."))
                if package.is_dir():
                    modules.extend(f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*")
            for module in modules:
                hits = module_files(module)
                if hits:
                    pending.extend(hits.difference(visited))
                elif (ROOT / module.split(".")[0]).exists():
                    unresolved.append(f"{path.relative_to(ROOT).as_posix()}:{module}")
    return sorted(visited), sorted(set(dynamic)), sorted(set(unresolved))


def year_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{year}-01-01 01:00", f"{year + 1}-01-01 00:00", freq="h", name="forecast_kst_dtm"
    )


def segments(year: int) -> dict[str, pd.DatetimeIndex]:
    make = lambda start, end: pd.date_range(start, end, freq="h", name="forecast_kst_dtm")
    return {
        "full": year_index(year),
        "H1": make(f"{year}-01-01 01:00", f"{year}-07-01 00:00"),
        "H2": make(f"{year}-07-01 01:00", f"{year + 1}-01-01 00:00"),
        "Q1": make(f"{year}-01-01 01:00", f"{year}-04-01 00:00"),
        "Q2": make(f"{year}-04-01 01:00", f"{year}-07-01 00:00"),
        "Q3": make(f"{year}-07-01 01:00", f"{year}-10-01 00:00"),
        "Q4": make(f"{year}-10-01 01:00", f"{year + 1}-01-01 00:00"),
    }


def read_prediction(path: Path, year: int) -> pd.DataFrame:
    frame = pd.read_parquet(path).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if tuple(frame.columns) != GROUPS or not frame.index.equals(year_index(year)):
        raise AssertionError(f"prediction contract differs: {path}")
    return frame


def bits(series: pd.Series) -> np.ndarray:
    return np.ascontiguousarray(series.to_numpy(np.float64)).view(np.uint64)


def assert_frame_bits(actual: pd.DataFrame, expected: pd.DataFrame, context: str) -> None:
    for group in GROUPS:
        if not np.array_equal(bits(actual[group]), bits(expected[group])):
            raise AssertionError(f"{context}/{group}: float64 bits differ")


def assert_nested_close(actual: Any, expected: Any, context: str) -> None:
    if isinstance(expected, Mapping):
        if set(actual) != set(expected):
            raise AssertionError(f"{context}: keys differ")
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{context}/{key}")
    elif isinstance(expected, bool):
        if actual is not expected:
            raise AssertionError(f"{context}: boolean differs")
    elif isinstance(expected, (int, np.integer)):
        if int(actual) != int(expected):
            raise AssertionError(f"{context}: integer differs")
    elif isinstance(expected, (float, np.floating)):
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=2e-14):
            raise AssertionError(f"{context}: float differs")
    elif actual != expected:
        raise AssertionError(f"{context}: value differs")


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
    rows = [group_metrics(actual[group], prediction[group], group) for group in GROUPS]
    one_minus = float(np.mean([row["one_minus_nmae"] for row in rows]))
    ficr = float(np.mean([row["ficr"] for row in rows]))
    return {"total_score": 0.5 * (one_minus + ficr), "one_minus_nmae": one_minus, "ficr": ficr}


def main() -> None:
    manifest_path = ARTIFACT / "manifest.json"
    before_canonical = {path.relative_to(ROOT).as_posix(): sha256(path) for path in ARTIFACT.rglob("*") if path.is_file()}
    if sha256(CONFIG) != CONFIG_SHA or sha256(manifest_path) != MANIFEST_SHA:
        raise AssertionError("config or manifest SHA differs")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar = CONFIG.with_suffix(".sha256")
    if sidecar.read_text(encoding="utf-8") != f"{CONFIG_SHA}  {CONFIG.name}\n":
        raise AssertionError("config sidecar differs")
    if config["lineage"]["scale097_raw_gpu_audit"]["sha256"] != PREVIOUS_AUDIT_SHA:
        raise AssertionError("prior independent audit lineage differs")
    for record in [*config["lineage"].values(), *config["input_identities"].values()]:
        verify_record(record)

    # Independently check the complete canonical output closure, not only listed records.
    listed: set[Path] = set()
    for record in manifest["outputs"]:
        listed.add(verify_record(record).resolve())
    actual = {path.resolve() for path in ARTIFACT.rglob("*") if path.is_file() and path != manifest_path}
    if listed != actual or len(actual) != int(manifest["output_count_excluding_manifest"]):
        raise AssertionError("manifest output closure differs")

    closure, dynamic, unresolved = ast_closure(ROOT / config["source_provenance_protocol"]["runner"])
    locked = manifest["source_provenance"]["recursive_AST_closure"]
    relative = [path.relative_to(ROOT).as_posix() for path in closure]
    if dynamic or unresolved or relative != locked["resolved_relative_paths"]:
        raise AssertionError("independent AST closure differs")
    locked_by_path = {Path(record["path"]).resolve(): record for record in locked["resolved_files"]}
    if set(closure) != set(locked_by_path) or len(closure) != 3:
        raise AssertionError("locked source set differs")
    for record in [*locked_by_path.values(), locked["test"], locked["config"], locked["sidecar"]]:
        verify_record(record)
    if locked["unresolved_local_imports"]:
        raise AssertionError("producer recorded unresolved local imports")

    # The two producer snapshots must be identical apart from phase/timestamp/assertion metadata,
    # and every one of the 42 protected upstream files must still match now.
    snapshot_before = json.loads((ARTIFACT / "protected_upstream_snapshot_before_metric_and_CSV.json").read_text(encoding="utf-8"))
    snapshot_after = json.loads((ARTIFACT / "protected_upstream_snapshot_after_metric_and_CSV.json").read_text(encoding="utf-8"))
    for key in ("schema_version", "protected_roots", "protected_files", "file_count", "files"):
        if snapshot_before[key] != snapshot_after[key]:
            raise AssertionError(f"protected snapshot differs: {key}")
    if snapshot_before["file_count"] != 42 or snapshot_after.get("exactly_equal_to_before") is not True:
        raise AssertionError("protected snapshot count/assertion differs")
    for record in snapshot_before["files"]:
        verify_record({"path": record["relative_path"], "size_bytes": record["size_bytes"], "sha256": record["sha256"]})

    identities = config["input_identities"]
    capacities = np.asarray([CAPACITY[group] for group in GROUPS], dtype=np.float64)

    # Rebuild 2024 A + D directly from the frozen upstream surfaces and compare float64 bits.
    input_A24 = read_prediction(verify_record(identities["A_2024_scale097"]), 2024)
    ref24 = read_prediction(verify_record(identities["ficr_delta_reference_2024"]), 2024)
    adjusted24 = read_prediction(verify_record(identities["ficr_delta_adjusted_2024"]), 2024)
    expected_D24 = adjusted24 - ref24
    expected_B24 = pd.DataFrame(
        np.clip(input_A24.to_numpy(np.float64) + expected_D24.to_numpy(np.float64), 0.0, 1.02 * capacities),
        index=input_A24.index,
        columns=GROUPS,
    )
    stored_A24 = read_prediction(ARTIFACT / "diagnostic_2024/A_scale097.parquet", 2024)
    stored_D24 = read_prediction(ARTIFACT / "diagnostic_2024/fixed_ficr_g1_delta.parquet", 2024)
    stored_B24 = read_prediction(ARTIFACT / "diagnostic_2024/B_scale097_plus_ficr_g1_delta.parquet", 2024)
    assert_frame_bits(stored_A24, input_A24, "A24")
    assert_frame_bits(stored_D24, expected_D24, "D24")
    assert_frame_bits(stored_B24, expected_B24, "B24")
    if np.count_nonzero(bits(stored_D24["kpx_group_2"])) or np.count_nonzero(bits(stored_D24["kpx_group_3"])):
        raise AssertionError("2024 G2/G3 delta is not positive-zero bit identity")
    nonzero24 = int(np.count_nonzero(stored_D24["kpx_group_1"].to_numpy()))
    if nonzero24 != 8148:
        raise AssertionError("2024 G1 delta support differs")

    labels = pd.read_csv(verify_record(identities["labels"]), encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    stored_metrics = json.loads((ARTIFACT / "interaction_diagnostic_2024.json").read_text(encoding="utf-8"))
    recomputed: dict[str, Any] = {}
    minimum_g1 = float("inf")
    minimum_mixed = float("inf")
    for segment, index in segments(2024).items():
        g1_a = group_metrics(labels.loc[index, "kpx_group_1"], stored_A24.loc[index, "kpx_group_1"], "kpx_group_1")
        g1_b = group_metrics(labels.loc[index, "kpx_group_1"], stored_B24.loc[index, "kpx_group_1"], "kpx_group_1")
        mixed_a = mixed_metrics(labels.loc[index, list(GROUPS)], stored_A24.loc[index])
        mixed_b = mixed_metrics(labels.loc[index, list(GROUPS)], stored_B24.loc[index])
        row = {
            "G1_A": g1_a,
            "G1_candidate": g1_b,
            "G1_delta_total_score": g1_b["total_score"] - g1_a["total_score"],
            "mixed_A": mixed_a,
            "mixed_candidate": mixed_b,
            "mixed_delta_total_score": mixed_b["total_score"] - mixed_a["total_score"],
            "mixed_delta_one_minus_nmae": mixed_b["one_minus_nmae"] - mixed_a["one_minus_nmae"],
            "mixed_delta_ficr": mixed_b["ficr"] - mixed_a["ficr"],
        }
        assert_nested_close(stored_metrics["comparisons"][segment], row, f"metrics/{segment}")
        recomputed[segment] = row
        minimum_g1 = min(minimum_g1, row["G1_delta_total_score"])
        minimum_mixed = min(minimum_mixed, row["mixed_delta_total_score"])
    full = recomputed["full"]
    go = all(row["G1_delta_total_score"] > 0.0 and row["mixed_delta_total_score"] > 0.0 for row in recomputed.values()) and full["mixed_delta_one_minus_nmae"] >= 0.0 and full["mixed_delta_ficr"] >= 0.0
    if go or stored_metrics["diagnostic_GO"] or stored_metrics["recommendation"] != "A_over_candidate":
        raise AssertionError("2024 diagnostic gate/recommendation differs")
    if manifest["diagnostic"] != stored_metrics:
        raise AssertionError("manifest diagnostic differs")

    # Rebuild final A + D from the direct composed Parquet and parsed reference CSV.
    input_A25 = read_prediction(verify_record(identities["A_final_scale097"]), 2025)
    reference_csv = pd.read_csv(verify_record(identities["ficr_delta_reference_final_csv"]), encoding="utf-8-sig")
    reference25 = reference_csv.loc[:, list(GROUPS)].astype(np.float64)
    reference25.index = year_index(2025)
    adjusted25 = read_prediction(verify_record(identities["ficr_composed_final_direct"]), 2025)
    expected_D25 = adjusted25 - reference25
    expected_B25 = pd.DataFrame(
        np.clip(input_A25.to_numpy(np.float64) + expected_D25.to_numpy(np.float64), 0.0, 1.02 * capacities),
        index=input_A25.index,
        columns=GROUPS,
    )
    stored_D25 = read_prediction(ARTIFACT / "final/fixed_ficr_g1_delta.parquet", 2025)
    stored_B25 = read_prediction(ARTIFACT / "final/scale097_plus_ficr_g1_delta.parquet", 2025)
    assert_frame_bits(stored_D25, expected_D25, "D25")
    assert_frame_bits(stored_B25, expected_B25, "B25")
    for group in ("kpx_group_2", "kpx_group_3"):
        if np.count_nonzero(bits(stored_D25[group])) or not np.array_equal(bits(stored_B25[group]), bits(input_A25[group])):
            raise AssertionError(f"final {group} identity differs")
    nonzero25 = int(np.count_nonzero(stored_D25["kpx_group_1"].to_numpy()))
    if nonzero25 != 8470:
        raise AssertionError("final G1 delta support differs")

    csv_path = ARTIFACT / config["output_contract"]["CSV"]
    raw = csv_path.read_bytes()
    if sha256(csv_path) != CSV_SHA or raw[:3] != b"\xef\xbb\xbf":
        raise AssertionError("CSV SHA/BOM differs")
    sample = pd.read_csv(verify_record(identities["sample"]), encoding="utf-8-sig", dtype="string")
    rendered = pd.read_csv(csv_path, encoding="utf-8-sig", dtype="string")
    if len(rendered) != 8760 or tuple(rendered.columns) != tuple(sample.columns):
        raise AssertionError("CSV schema/rows differs")
    if not rendered[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]]):
        raise AssertionError("CSV IDs/timestamps differ")
    for group in GROUPS:
        expected_text = stored_B25[group].map(lambda value: f"{float(value):.6f}").reset_index(drop=True).astype("string")
        if not rendered[group].equals(expected_text):
            raise AssertionError(f"CSV six-decimal roundtrip differs: {group}")
    numeric = rendered.loc[:, list(GROUPS)].astype(np.float64).to_numpy()
    if not np.isfinite(numeric).all() or np.any(numeric < 0.0) or np.any(numeric > 1.02 * capacities + 5e-7):
        raise AssertionError("CSV finite/bounds differs")

    after_canonical = {path.relative_to(ROOT).as_posix(): sha256(path) for path in ARTIFACT.rglob("*") if path.is_file()}
    if before_canonical != after_canonical:
        raise AssertionError("canonical changed during audit")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "read_only": True,
        "artifact": "public_adaptive_scale097_ficr_g1_delta_v1",
        "config_sha256": CONFIG_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "independent_AST": {"exact": True, "files": relative, "dynamic_imports": dynamic, "unresolved_local_imports": unresolved},
        "manifest_output_and_source_closure_exact": True,
        "test": {"sha256": locked["test"]["sha256"], "focused_passed": 8, "focused_failed": 0},
        "protected_upstream": {"snapshot_before_after_exact": True, "current_exact": True, "file_count": 42},
        "final": {
            "A_plus_D_direct_formula_exact": True,
            "G1_delta_nonzero_rows": nonzero25,
            "G2_G3_A_bit_identity": True,
            "CSV_sha256": CSV_SHA,
            "CSV_contracts_exact": True,
        },
        "diagnostic_2024": {
            "A_plus_D_direct_formula_exact": True,
            "G1_delta_nonzero_rows": nonzero24,
            "G2_G3_A_bit_identity": True,
            "metric_records_recomputed": 7,
            "minimum_G1_delta": minimum_g1,
            "minimum_mixed_delta": minimum_mixed,
            "GO": go,
            "recommendation": "A_over_candidate",
        },
        "risk": manifest["risk"],
        "canonical_nonmutation": True,
    }
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_name(f".{DESTINATION.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, DESTINATION)
    print(f"PASS {DESTINATION} sha256={sha256(DESTINATION)}")


if __name__ == "__main__":
    main()

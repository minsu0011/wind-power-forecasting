"""Independent, read-only audit of the G2 post-hoc rescue v5 artifact."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CANONICAL = ROOT / "artifacts/postgate/direct_interval_g2_posthoc_rescue_v5"
QUARANTINED_V4 = (
    ROOT
    / "artifacts/postgate/_failed_provenance_coverage/direct_interval_g2_posthoc_rescue_v4"
)
V2 = ROOT / "artifacts/postgate/direct_interval_selective_scale_transfer_v2"
LABELS = Path(r"data/local/open/train/train_labels.csv")
SAMPLE = Path(r"data/local/open/sample_submission.csv")
GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
EXPECTED = {
    "config": "a0f847c980bd2694e7d76b5258a7d19138d79d537f1fcdc6dc8c0739bc0f8dd1",
    "closure_lock": "172b410979b0f681706deb4b444c36969b4fa23d1f67d875dd1bda8887bc287c",
    "prescore_lock": "bec01b08c3269089fc165376aab6901c7a2f077ee2bd4f7f674929f00c24596b",
    "manifest": "e02b966a0e02e06a8d12fc4ce23b129bde6c7f0f33e870231537b0b938d97f10",
    "csv": "2596bf1048f7e1635c7c965096f896d5a4104f3e04b58f575e21f9e2314bfddf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _file_record_exact(record: Mapping[str, Any]) -> None:
    path = Path(str(record["path"]))
    expected_size = int(record.get("size_bytes", record.get("bytes")))
    if not path.is_file() or path.stat().st_size != expected_size:
        raise AssertionError(f"missing/size mismatch: {path}")
    if sha256(path) != str(record["sha256"]):
        raise AssertionError(f"hash mismatch: {path}")


def _verify_registered_files(node: Any) -> int:
    count = 0
    if isinstance(node, Mapping):
        if "path" in node and "sha256" in node and (
            "bytes" in node or "size_bytes" in node
        ):
            _file_record_exact(node)
            count += 1
        for value in node.values():
            count += _verify_registered_files(value)
    elif isinstance(node, list):
        for value in node:
            count += _verify_registered_files(value)
    return count


def _module_file(module: str) -> Path | None:
    base = ROOT.joinpath(*module.split("."))
    file_path = base.with_suffix(".py")
    if file_path.is_file():
        return file_path.resolve()
    init = base / "__init__.py"
    if init.is_file():
        return init.resolve()
    return None


def _package_initializers(path: Path) -> set[Path]:
    result: set[Path] = set()
    parent = path.parent
    while parent != ROOT and ROOT in parent.parents:
        init = parent / "__init__.py"
        if init.is_file():
            result.add(init.resolve())
        parent = parent.parent
    return result


def independent_local_import_closure(entrypoint: Path) -> tuple[set[Path], list[str]]:
    """Resolve local imports without consulting the producer lock or manifest."""

    pending = [entrypoint.resolve()]
    resolved: set[Path] = set()
    dynamic: list[str] = []
    while pending:
        path = pending.pop()
        if path in resolved:
            continue
        resolved.add(path)
        for init in _package_initializers(path):
            if init not in resolved:
                pending.append(init)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {
                    "__import__",
                    "exec",
                    "eval",
                }:
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
                package_dir = ROOT.joinpath(*node.module.split("."))
                if package_dir.is_dir():
                    modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
            for module in modules:
                candidate = _module_file(module)
                if candidate is not None and candidate not in resolved:
                    pending.append(candidate)
    return resolved, dynamic


def year_segments(year: int) -> dict[str, pd.DatetimeIndex]:
    start = pd.Timestamp(f"{year}-01-01 01:00")
    q2 = pd.Timestamp(f"{year}-04-01 01:00")
    h2 = pd.Timestamp(f"{year}-07-01 01:00")
    q4 = pd.Timestamp(f"{year}-10-01 01:00")
    end = pd.Timestamp(f"{year + 1}-01-01 00:00")
    hour = pd.Timedelta(hours=1)
    make = lambda a, b: pd.date_range(a, b, freq="h", name="forecast_kst_dtm")
    return {
        "full": make(start, end),
        "H1": make(start, h2 - hour),
        "H2": make(h2, end),
        "Q1": make(start, q2 - hour),
        "Q2": make(q2, h2 - hour),
        "Q3": make(h2, q4 - hour),
        "Q4": make(q4, end),
    }


def group_metrics(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    capacity = CAPACITY[group]
    y = actual.to_numpy(np.float64)
    p = prediction.to_numpy(np.float64)
    valid = np.isfinite(y) & (y >= 0.10 * capacity)
    y, p = y[valid], p[valid]
    error = np.abs(p - y) / capacity
    one_minus = 1.0 - float(np.mean(error))
    unit = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    ficr = float(np.sum(y * unit) / np.sum(y * 4.0))
    return {
        "score": 0.5 * (one_minus + ficr),
        "one_minus_nmae": one_minus,
        "ficr": ficr,
        "n": int(valid.sum()),
    }


def mixed_metrics(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, float]:
    records = [group_metrics(actual[g], prediction[g], g) for g in GROUPS]
    one_minus = float(np.mean([row["one_minus_nmae"] for row in records]))
    ficr = float(np.mean([row["ficr"] for row in records]))
    return {
        "total_score": 0.5 * (one_minus + ficr),
        "one_minus_nmae": one_minus,
        "ficr": ficr,
    }


def interpolate(surface: np.ndarray, query: np.ndarray) -> np.ndarray:
    coordinate = 100.0 * np.clip(query, 0.0, 1.02)
    lower = np.floor(coordinate).astype(np.int64)
    upper = np.minimum(lower + 1, 102)
    fraction = coordinate - lower
    rows = np.arange(len(query))
    return (1.0 - fraction) * surface[rows, lower] + fraction * surface[rows, upper]


def verify_formula(
    baseline: pd.Series,
    candidate: pd.Series,
    diagnostics: pd.DataFrame,
    utility: np.ndarray,
) -> int:
    capacity = CAPACITY["kpx_group_2"]
    base = baseline.to_numpy(np.float64)
    base_cf = base / capacity
    scaled_cf = 0.98 * base_cf
    base_utility = interpolate(utility, base_cf)
    scaled_utility = interpolate(utility, scaled_cf)
    advantage = scaled_utility - base_utility
    gate = advantage > 0.01
    expected = np.clip(np.where(gate, 0.98 * base, base), 0.0, 1.02 * capacity)
    if not np.array_equal(candidate.to_numpy(np.float64), expected):
        raise AssertionError("candidate formula differs")
    columns = {
        "baseline_cf": base_cf,
        "scaled_cf": scaled_cf,
        "base_utility": base_utility,
        "scaled_utility": scaled_utility,
        "utility_advantage": advantage,
        "gate": gate,
        "candidate_cf": expected / capacity,
    }
    if not diagnostics.index.equals(baseline.index):
        raise AssertionError("diagnostic index differs")
    for column, values in columns.items():
        if not np.array_equal(diagnostics[column].to_numpy(), values, equal_nan=True):
            raise AssertionError(f"diagnostic differs: {column}")
    return int(gate.sum())


def bits(series: pd.Series) -> np.ndarray:
    return np.ascontiguousarray(series.to_numpy(np.float64)).view(np.uint64)


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    return frame


def main() -> None:
    before = {
        "v5_manifest": sha256(CANONICAL / "manifest.json"),
        "v4_manifest": sha256(QUARANTINED_V4 / "manifest.json"),
        "v2_manifest": sha256(V2 / "manifest.json"),
    }
    if before["v5_manifest"] != EXPECTED["manifest"]:
        raise AssertionError("v5 manifest hash differs")
    if sha256(CANONICAL / "preregister.json") != EXPECTED["config"]:
        raise AssertionError("config hash differs")
    if sha256(CANONICAL / "source_closure_before_fit_lock.json") != EXPECTED["closure_lock"]:
        raise AssertionError("source closure lock hash differs")
    if sha256(CANONICAL / "final_prescore_lock.json") != EXPECTED["prescore_lock"]:
        raise AssertionError("prescore lock hash differs")
    csv_path = CANONICAL / "direct_interval_g2_posthoc_rescue_v5_2025.csv"
    if sha256(csv_path) != EXPECTED["csv"]:
        raise AssertionError("CSV hash differs")

    manifest = json.loads((CANONICAL / "manifest.json").read_text(encoding="utf-8"))
    listed: set[Path] = set()
    for record in manifest["outputs"]:
        _file_record_exact(record)
        listed.add(Path(record["path"]).resolve())
    actual = {
        path.resolve()
        for path in CANONICAL.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if listed != actual or len(actual) != int(manifest["output_count_excluding_manifest"]):
        raise AssertionError("manifest output closure differs")

    closure_lock = json.loads(
        (CANONICAL / "source_closure_before_fit_lock.json").read_text(encoding="utf-8")
    )
    locked_closure = closure_lock["closure"]
    independently_resolved, dynamic_imports = independent_local_import_closure(
        ROOT / str(locked_closure["entrypoint"])
    )
    independent_relative = sorted(path.relative_to(ROOT).as_posix() for path in independently_resolved)
    if dynamic_imports:
        raise AssertionError(f"dynamic imports prevent complete closure: {dynamic_imports}")
    if independent_relative != list(locked_closure["resolved_relative_paths"]):
        raise AssertionError("independently resolved AST closure differs")
    locked_files = {Path(row["path"]).resolve(): row for row in locked_closure["resolved_files"]}
    if independently_resolved != set(locked_files):
        raise AssertionError("closure file set differs")
    for record in locked_files.values():
        _file_record_exact(record)
    if locked_closure["resolved_file_count"] != len(independently_resolved):
        raise AssertionError("closure count differs")
    if locked_closure["unresolved_local_imports"]:
        raise AssertionError("producer recorded unresolved local imports")
    if manifest["source_provenance"]["recursive_AST_local_import_closure"] != locked_closure:
        raise AssertionError("manifest source closure record differs from pre-fit lock")
    if _verify_registered_files(manifest["source_provenance"]) < 9:
        raise AssertionError("source/config/test/incident records unexpectedly incomplete")

    config = json.loads((CANONICAL / "preregister.json").read_text(encoding="utf-8"))
    registered_inputs = _verify_registered_files(config)
    if registered_inputs < 17:
        raise AssertionError("registered input coverage unexpectedly small")
    risk = manifest["risk"]
    if not risk["selection_unsafe"] or risk["strict_final_isolation"]:
        raise AssertionError("risk disclosure differs")
    if risk["private_champion"] or risk["leaderboard_score_claim"]:
        raise AssertionError("unsupported performance claim")

    labels = pd.read_csv(
        LABELS, encoding="utf-8-sig", parse_dates=["kst_dtm"]
    ).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    labels_2024 = labels.loc[year_segments(2024)["full"], list(GROUPS)]
    results = json.loads((CANONICAL / "rescue_2024_results.json").read_text(encoding="utf-8"))
    v4_config = json.loads(
        (ROOT / "configs/direct_interval_g2_posthoc_rescue_preregister_v4.json").read_text(
            encoding="utf-8"
        )
    )
    frozen = v4_config["frozen_2024_posthoc_rescue_evidence"]
    with np.load(V2 / "stage2/surfaces/kpx_group_2.npz") as payload:
        utility_2024 = payload["utility"]
    stage2_report: dict[str, Any] = {}
    for name in ("primary_v3", "interaction_v4"):
        baseline = _read_frame(CANONICAL / f"rescue_2024/{name}/baseline.parquet")
        candidate = _read_frame(CANONICAL / f"rescue_2024/{name}/candidate.parquet")
        diagnostics = pd.read_parquet(
            CANONICAL / f"rescue_2024/{name}/g2_diagnostics.parquet"
        )
        v2_baseline = _read_frame(V2 / f"stage2/{name}/baseline_2024.parquet")
        v2_candidate = _read_frame(V2 / f"stage2/{name}/candidate_2024.parquet")
        v2_diagnostics = pd.read_parquet(
            V2 / f"stage2/{name}/kpx_group_2_diagnostics.parquet"
        )
        for group in GROUPS:
            if not np.array_equal(bits(baseline[group]), bits(v2_baseline[group])):
                raise AssertionError(f"{name}: baseline differs from v2")
        if not np.array_equal(bits(candidate["kpx_group_2"]), bits(v2_candidate["kpx_group_2"])):
            raise AssertionError(f"{name}: G2 candidate differs from v2")
        if not diagnostics.equals(v2_diagnostics):
            raise AssertionError(f"{name}: diagnostics differ from v2")
        for group in ("kpx_group_1", "kpx_group_3"):
            if not np.array_equal(bits(candidate[group]), bits(baseline[group])):
                raise AssertionError(f"{name}: {group} is not bit identity")
        gates = verify_formula(baseline["kpx_group_2"], candidate["kpx_group_2"], diagnostics, utility_2024)
        stored = results["baseline_results"][name]
        group_deltas: list[float] = []
        mixed_deltas: list[float] = []
        for segment, index in year_segments(2024).items():
            gb = group_metrics(labels_2024.loc[index, "kpx_group_2"], baseline.loc[index, "kpx_group_2"], "kpx_group_2")
            gc = group_metrics(labels_2024.loc[index, "kpx_group_2"], candidate.loc[index, "kpx_group_2"], "kpx_group_2")
            mb = mixed_metrics(labels_2024.loc[index], baseline.loc[index])
            mc = mixed_metrics(labels_2024.loc[index], candidate.loc[index])
            gd = gc["score"] - gb["score"]
            md = mc["total_score"] - mb["total_score"]
            assert_close(stored["g2_group"][segment], {"baseline": gb, "candidate": gc, "delta": gd}, f"{name}/g2/{segment}")
            assert_close(stored["mixed"][segment], {"baseline": mb, "candidate": mc, "delta_total_score": md}, f"{name}/mixed/{segment}")
            assert_close(gd, frozen["g2_group_delta_total_score"][name][segment], f"{name}/frozen-g2/{segment}")
            assert_close(md, frozen["mixed_delta_total_score"][name][segment], f"{name}/frozen-mixed/{segment}")
            group_deltas.append(gd)
            mixed_deltas.append(md)
        full = stored["mixed"]["full"]
        delta_nmae = full["candidate"]["one_minus_nmae"] - full["baseline"]["one_minus_nmae"]
        delta_ficr = full["candidate"]["ficr"] - full["baseline"]["ficr"]
        assert_close(delta_nmae, frozen["mixed_full_component_deltas"][name]["one_minus_nmae"], f"{name}/full-nmae")
        assert_close(delta_ficr, frozen["mixed_full_component_deltas"][name]["ficr"], f"{name}/full-ficr")
        passed = all(value > 0.0 for value in group_deltas + mixed_deltas) and delta_nmae >= 0.0 and delta_ficr >= 0.0
        if passed != bool(stored["all_gates_passed"]):
            raise AssertionError(f"{name}: gate result differs")
        stage2_report[name] = {
            "formula_exact": True,
            "selected_rows": gates,
            "metric_records_recomputed": 14,
            "minimum_g2_delta": min(group_deltas),
            "minimum_mixed_delta": min(mixed_deltas),
            "full_delta_one_minus_nmae": delta_nmae,
            "full_delta_ficr": delta_ficr,
            "passed": passed,
        }

    from src.direct_interval_probability import CONTEXT_COLUMNS

    baseline_final = _read_frame(
        ROOT / "artifacts/final_cf_fix/predictions/corrected_recent_v4_test.parquet"
    )
    prediction = _read_frame(CANONICAL / "final/predictions.parquet")
    diagnostics = pd.read_parquet(CANONICAL / "final/g2_diagnostics.parquet")
    context = pd.read_parquet(
        ROOT / "artifacts/cache/kpx_group_2_weather_test.parquet",
        columns=list(CONTEXT_COLUMNS),
    )
    context.index = pd.DatetimeIndex(context.index, name="forecast_kst_dtm")
    context = context.astype(np.float32)
    model = joblib.load(CANONICAL / "final/model/kpx_group_2.joblib")
    raw_surface = model.predict_surfaces(context)
    replay = {
        key: np.asarray(raw_surface[key], dtype=np.float64)
        for key in ("p6_raw", "p8_raw", "p6", "p8", "eae")
    }
    replay["utility"] = -0.5 * np.clip(replay["eae"], 0.0, 1.20) + 0.5 * (
        0.25 * replay["p6"] + 0.75 * replay["p8"]
    )
    replay["repair_count"] = np.asarray([raw_surface["repair_count"]], dtype=np.int64)
    with np.load(CANONICAL / "final/surface/kpx_group_2.npz") as saved:
        if set(saved.files) != set(replay) or any(
            not np.array_equal(saved[key], replay[key]) for key in replay
        ):
            raise AssertionError("final model-to-surface bit replay differs")
    final_gates = verify_formula(
        baseline_final["kpx_group_2"],
        prediction["kpx_group_2"],
        diagnostics,
        replay["utility"],
    )
    if final_gates != 143:
        raise AssertionError("final selected-row count differs")
    for group in ("kpx_group_1", "kpx_group_3"):
        if not np.array_equal(bits(prediction[group]), bits(baseline_final[group])):
            raise AssertionError(f"final {group} is not bit identity")
    expected_index = pd.date_range(
        "2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm"
    )
    if len(prediction) != 8760 or not prediction.index.equals(expected_index):
        raise AssertionError("final prediction rows/time differ")

    output = pd.read_csv(csv_path, encoding="utf-8-sig", dtype="string")
    sample = pd.read_csv(SAMPLE, encoding="utf-8-sig", dtype="string")
    if not csv_path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise AssertionError("CSV BOM missing")
    if tuple(output.columns) != tuple(sample.columns) or len(output) != 8760:
        raise AssertionError("CSV schema/rows differ")
    if not output[["forecast_id", "forecast_kst_dtm"]].equals(
        sample[["forecast_id", "forecast_kst_dtm"]]
    ):
        raise AssertionError("CSV identifiers/timestamps differ")
    for group in GROUPS:
        expected_text = prediction[group].map(lambda value: f"{float(value):.6f}").astype("string")
        if not output[group].equals(expected_text.reset_index(drop=True)):
            raise AssertionError(f"CSV six-decimal roundtrip differs: {group}")
    numeric = output.loc[:, list(GROUPS)].astype(np.float64).to_numpy()
    upper = 1.02 * np.asarray([CAPACITY[group] for group in GROUPS])
    if not np.isfinite(numeric).all() or np.any(numeric < 0.0) or np.any(numeric > upper + 5e-7):
        raise AssertionError("CSV finite/capacity bound check differs")

    v4_prediction = _read_frame(QUARANTINED_V4 / "final/predictions.parquet")
    v4_diagnostics = pd.read_parquet(QUARANTINED_V4 / "final/g2_diagnostics.parquet")
    for group in GROUPS:
        if not np.array_equal(bits(prediction[group]), bits(v4_prediction[group])):
            raise AssertionError(f"v4-v5 prediction bits differ: {group}")
    if not np.array_equal(
        diagnostics["gate"].to_numpy(), v4_diagnostics["gate"].to_numpy()
    ):
        raise AssertionError("v4-v5 G2 gate booleans differ")
    v4_csv = QUARANTINED_V4 / "direct_interval_g2_posthoc_rescue_v4_2025.csv"
    if csv_path.read_bytes() != v4_csv.read_bytes():
        raise AssertionError("v4-v5 CSV bytes differ")

    prescore = json.loads((CANONICAL / "final_prescore_lock.json").read_text(encoding="utf-8"))
    final = json.loads((CANONICAL / "final_results.json").read_text(encoding="utf-8"))
    if any(
        int(prescore[key]) != 0
        for key in (
            "test_context_values_parsed_before_prescore_lock",
            "recent_v4_baseline_values_parsed_before_prescore_lock",
            "sample_values_parsed_before_prescore_lock",
        )
    ):
        raise AssertionError("pre-score access counters differ")
    if prescore["model_copied_from_v4"] or prescore["candidate_rule_changed_from_v4"]:
        raise AssertionError("fresh-fit/same-rule declaration differs")
    if not (
        closure_lock["created_utc"]
        <= results["created_utc"]
        <= prescore["created_utc"]
        <= final["created_utc"]
        <= manifest["created_utc"]
    ):
        raise AssertionError("recorded execution ordering differs")

    after = {
        "v5_manifest": sha256(CANONICAL / "manifest.json"),
        "v4_manifest": sha256(QUARANTINED_V4 / "manifest.json"),
        "v2_manifest": sha256(V2 / "manifest.json"),
    }
    if before != after:
        raise AssertionError("canonical artifacts changed during read-only audit")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "artifact": "direct_interval_g2_posthoc_rescue_v5",
        "hashes": {**EXPECTED, "prediction": sha256(CANONICAL / "final/predictions.parquet")},
        "manifest": {
            "output_closure_exact": True,
            "output_count_excluding_manifest": len(actual),
            "registered_input_records_verified": registered_inputs,
        },
        "source_provenance": {
            "independent_recursive_AST_closure_exact": True,
            "dynamic_imports": dynamic_imports,
            "resolved_file_count": len(independently_resolved),
            "resolved_relative_paths": independent_relative,
            "hashes_and_sizes_exact": True,
            "locked_before_fit_and_unchanged": True,
        },
        "stage2_2024": stage2_report,
        "final_2025": {
            "model_surface_bit_replay_exact": True,
            "factor": 0.98,
            "margin": 0.01,
            "lookup": "linear",
            "comparison": "strict_greater_than",
            "selected_rows": final_gates,
            "G1_G3_parquet_float64_identity_bits": True,
            "rows_and_time_exact": True,
            "csv_sha256": EXPECTED["csv"],
            "csv_BOM_schema_sample_ids_time_six_decimal_finite_bounds": True,
        },
        "supersession": {
            "v4_quarantined": True,
            "all_three_prediction_float64_bits_exact": True,
            "G2_gate_boolean_exact": True,
            "CSV_bytes_exact": True,
        },
        "risk_and_incident_disclosure": True,
        "recorded_order_and_zero_access_counters": True,
        "canonical_nonmutation": True,
    }
    destination = ROOT / "artifacts/audits/direct_interval_g2_posthoc_rescue_v5_independent.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(f"PASS {destination} sha256={sha256(destination)}")


if __name__ == "__main__":
    main()

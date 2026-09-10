"""Fresh provenance-complete supersession of the frozen G2-only rescue."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.direct_interval_probability import (  # noqa: E402
    COMMON_PARAMETERS,
    CONTEXT_COLUMNS,
    DirectIntervalProbabilityModel,
    official_action_utility,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


CONFIG_SHA = "a0f847c980bd2694e7d76b5258a7d19138d79d537f1fcdc6dc8c0739bc0f8dd1"
GROUP = "kpx_group_2"
IDENTITY_GROUPS = ("kpx_group_1", "kpx_group_3")
SPEC = (0.98, 0.01)
SEGMENT_NAMES = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
BASELINES = ("primary_v3", "interaction_v4")
V4_EXPECTED_CSV_SHA = "2596bf1048f7e1635c7c965096f896d5a4104f3e04b58f575e21f9e2314bfddf"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prescore", "final"), required=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/direct_interval_g2_posthoc_rescue_preregister_v5.json"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("artifacts/postgate/direct_interval_g2_posthoc_rescue_v5"),
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


def _copy_exclusive(source: Path, destination: Path) -> None:
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


def _atomic_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, **arrays)
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


def _verify_file(spec: Mapping[str, Any]) -> Path:
    path = _absolute(spec)
    if not path.is_file():
        raise FileNotFoundError(path)
    registered_size = spec.get("bytes", spec.get("size_bytes"))
    if registered_size is not None and path.stat().st_size != int(registered_size):
        raise AssertionError(f"size differs: {path}")
    if sha256_file(path) != str(spec["sha256"]):
        raise AssertionError(f"hash differs: {path}")
    return path


def _verify_config(path: Path) -> dict[str, Any]:
    if sha256_file(path) != CONFIG_SHA:
        raise AssertionError("v5 preregistration hash differs")
    expected_sidecar = f"{CONFIG_SHA}  {path.name}\n"
    if path.with_suffix(".sha256").read_text(encoding="utf-8") != expected_sidecar:
        raise AssertionError("v5 preregistration sidecar differs")
    config = json.loads(path.read_text(encoding="utf-8"))
    rule = config["immutable_candidate"]
    g2 = rule[GROUP]
    if (float(g2["factor"]), float(g2["margin"])) != SPEC:
        raise AssertionError("candidate rule differs")
    if not all(rule[group].get("identity") is True for group in IDENTITY_GROUPS):
        raise AssertionError("identity rule differs")
    if config["risk_classification"]["selection_unsafe"] is not True:
        raise AssertionError("risk disclosure differs")
    for section in ("supersession", "frozen_2024_evidence", "input_identities"):
        for value in config[section].values():
            if isinstance(value, Mapping) and "path" in value and "sha256" in value:
                _verify_file(value)
    return config


def _module_files(module: str) -> list[Path]:
    if not module:
        return []
    parts = module.split(".")
    file_candidate = PROJECT_DIR.joinpath(*parts).with_suffix(".py")
    package_candidate = PROJECT_DIR.joinpath(*parts, "__init__.py")
    found = []
    if file_candidate.is_file():
        found.append(file_candidate.resolve())
    if package_candidate.is_file():
        found.append(package_candidate.resolve())
    # Importing a nested local module executes each extant package initializer.
    for depth in range(1, len(parts)):
        initializer = PROJECT_DIR.joinpath(*parts[:depth], "__init__.py")
        if initializer.is_file():
            found.append(initializer.resolve())
    return sorted(set(found))


def resolve_ast_local_import_closure(entrypoint: Path) -> tuple[Path, ...]:
    entry = entrypoint.resolve()
    if PROJECT_DIR not in entry.parents:
        raise AssertionError("entrypoint escapes project")
    queue = [entry]
    visited: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = _module_files(alias.name)
                    if resolved:
                        queue.extend(resolved)
                    elif (PROJECT_DIR / alias.name.split(".")[0]).exists():
                        raise AssertionError(f"unresolved local import: {alias.name} from {path}")
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative = path.relative_to(PROJECT_DIR).with_suffix("").parts[:-1]
                    keep = len(relative) - node.level + 1
                    prefix = relative[:max(keep, 0)]
                    base = ".".join((*prefix, *(node.module or "").split(".")))
                else:
                    base = node.module or ""
                resolved_base = _module_files(base.strip("."))
                if resolved_base:
                    queue.extend(resolved_base)
                else:
                    top = base.strip(".").split(".")[0]
                    if top and (PROJECT_DIR / top).exists():
                        raise AssertionError(f"unresolved local import: {base} from {path}")
                # A from-import name can be either a symbol or a submodule. Resolve
                # the submodule when present, but a normal symbol is not an error.
                for alias in node.names:
                    if alias.name != "*":
                        queue.extend(_module_files(f"{base}.{alias.name}".strip(".")))
    return tuple(sorted(visited))


def _closure_record(args: argparse.Namespace) -> dict[str, Any]:
    closure = resolve_ast_local_import_closure(Path(__file__))
    relative = [path.relative_to(PROJECT_DIR).as_posix() for path in closure]
    test_path = PROJECT_DIR / "tests/test_direct_interval_g2_posthoc_rescue_v5.py"
    incident_path = PROJECT_DIR / "configs/direct_interval_g2_v4_provenance_incident_20260808.json"
    record = {
        "resolver": "recursive Python AST local imports plus package initializers",
        "entrypoint": Path(__file__).resolve().relative_to(PROJECT_DIR).as_posix(),
        "resolved_relative_paths": relative,
        "resolved_file_count": len(closure),
        "resolved_files": [describe_file(path) for path in closure],
        "unresolved_local_imports": [],
        "validation_source": describe_file(test_path),
        "config": describe_file(args.config.resolve()),
        "config_sidecar": describe_file(args.config.with_suffix(".sha256").resolve()),
        "incident": describe_file(incident_path),
        "incident_sidecar": describe_file(incident_path.with_suffix(".sha256")),
    }
    return record


def _assert_closure_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    keys = (
        "resolver", "entrypoint", "resolved_relative_paths", "resolved_file_count",
        "resolved_files", "unresolved_local_imports", "validation_source", "config",
        "config_sidecar", "incident", "incident_sidecar",
    )
    for key in keys:
        if first[key] != second[key]:
            raise AssertionError(f"source closure changed: {key}")


def _year_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{year}-01-01 01:00", f"{year + 1}-01-01 00:00", freq="h",
        name="forecast_kst_dtm",
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


def _read_labels(spec: Mapping[str, Any]) -> pd.DataFrame:
    path = _verify_file(spec)
    frame = pd.read_csv(path)
    if tuple(frame.columns) != ("kst_dtm", *TARGET_COLS) or len(frame) != int(spec["rows"]):
        raise AssertionError("label schema/rows differ")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm")), name="forecast_kst_dtm")
    if not frame.index.equals(pd.date_range("2022-01-01 01:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm")):
        raise AssertionError("label index differs")
    return frame.astype(np.float64)


def _read_context(spec: Mapping[str, Any], index: pd.DatetimeIndex) -> pd.DataFrame:
    path = _verify_file(spec)
    frame = pd.read_parquet(path, columns=list(CONTEXT_COLUMNS))
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index) or tuple(frame.columns) != CONTEXT_COLUMNS:
        raise AssertionError("context schema/index differs")
    frame = frame.astype(np.float32)
    if not np.isfinite(frame.to_numpy()).all():
        raise AssertionError("context contains non-finite values")
    return frame


def _bits(series: pd.Series) -> np.ndarray:
    return np.ascontiguousarray(series.to_numpy(dtype=np.float64)).view(np.uint64)


def _assert_identity(candidate: pd.DataFrame, baseline: pd.DataFrame) -> None:
    if not candidate.index.equals(baseline.index):
        raise AssertionError("candidate index differs")
    for group in IDENTITY_GROUPS:
        if not np.array_equal(_bits(candidate[group]), _bits(baseline[group])):
            raise AssertionError(f"identity bits differ: {group}")


def interpolate_utility(utility: np.ndarray, query_cf: np.ndarray) -> np.ndarray:
    q = 100.0 * np.clip(np.asarray(query_cf, dtype=np.float64), 0.0, 1.02)
    lower = np.floor(q).astype(np.int64)
    upper = np.minimum(lower + 1, 102)
    fraction = q - lower
    row = np.arange(len(q), dtype=np.int64)
    return (1.0 - fraction) * utility[row, lower] + fraction * utility[row, upper]


def apply_rule(baseline: pd.Series, utility: np.ndarray) -> tuple[pd.Series, pd.DataFrame]:
    values = baseline.to_numpy(dtype=np.float64)
    baseline_cf = values / CAPACITY_KWH[GROUP]
    base_utility = interpolate_utility(utility, baseline_cf)
    scaled_cf = SPEC[0] * baseline_cf
    scaled_utility = interpolate_utility(utility, scaled_cf)
    advantage = scaled_utility - base_utility
    gate = advantage > SPEC[1]
    candidate_values = np.clip(
        np.where(gate, SPEC[0] * values, values), 0.0,
        1.02 * CAPACITY_KWH[GROUP],
    )
    candidate = pd.Series(candidate_values, index=baseline.index, name=baseline.name)
    detail = pd.DataFrame(
        {
            "baseline_cf": baseline_cf,
            "scaled_cf": scaled_cf,
            "base_utility": base_utility,
            "scaled_utility": scaled_utility,
            "utility_advantage": advantage,
            "gate": gate,
            "candidate_cf": candidate_values / CAPACITY_KWH[GROUP],
        },
        index=baseline.index,
    )
    detail.index.name = "forecast_kst_dtm"
    return candidate, detail


def _group_metrics(actual: pd.Series, forecast: pd.Series, group: str) -> dict[str, Any]:
    capacity = CAPACITY_KWH[group]
    y = actual.to_numpy(dtype=np.float64)
    p = forecast.to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & (y >= 0.10 * capacity)
    y, p = y[valid], p[valid]
    error = np.abs(p - y) / capacity
    n = 1.0 - float(error.mean())
    price = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    f = float(np.sum(y * price) / np.sum(y * 4.0))
    return {"score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f, "n": int(valid.sum())}


def _mixed_metrics(actual: pd.DataFrame, forecast: pd.DataFrame) -> dict[str, float]:
    records = [_group_metrics(actual[group], forecast[group], group) for group in TARGET_COLS]
    n = float(np.mean([item["one_minus_nmae"] for item in records]))
    f = float(np.mean([item["ficr"] for item in records]))
    return {"total_score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f}


def _surface_arrays(model: DirectIntervalProbabilityModel, context: pd.DataFrame) -> dict[str, np.ndarray]:
    raw = model.predict_surfaces(context)
    arrays = {
        "p6_raw": np.asarray(raw["p6_raw"], dtype=np.float64),
        "p8_raw": np.asarray(raw["p8_raw"], dtype=np.float64),
        "p6": np.asarray(raw["p6"], dtype=np.float64),
        "p8": np.asarray(raw["p8"], dtype=np.float64),
        "eae": np.asarray(raw["eae"], dtype=np.float64),
        "repair_count": np.asarray([raw["repair_count"]], dtype=np.int64),
    }
    arrays["utility"] = official_action_utility(arrays["p6"], arrays["p8"], arrays["eae"])
    return arrays


def _assert_arrays(first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]) -> None:
    if set(first) != set(second):
        raise AssertionError("surface keys differ")
    for key in first:
        if not np.array_equal(first[key], second[key]):
            raise AssertionError(f"surface differs: {key}")


def _rebuild_2024(config: Mapping[str, Any], out_dir: Path, labels: pd.DataFrame) -> dict[str, Any]:
    v4_config = json.loads(_verify_file(config["supersession"]["v4_preregister"]).read_text(encoding="utf-8"))
    expected = v4_config["frozen_2024_posthoc_rescue_evidence"]
    registered = config["frozen_2024_evidence"]
    with np.load(_verify_file(registered["v2_g2_surface"])) as payload:
        utility = np.asarray(payload["utility"], dtype=np.float64)
    segments = _segments(2024)
    result: dict[str, Any] = {}
    for name in BASELINES:
        prefix = "primary_v3" if name == "primary_v3" else "interaction_v4"
        baseline = pd.read_parquet(_verify_file(registered[f"{prefix}_baseline"])).astype(np.float64)
        original = pd.read_parquet(_verify_file(registered[f"{prefix}_v2_candidate"])).astype(np.float64)
        for frame in (baseline, original):
            frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        selected, detail = apply_rule(baseline[GROUP], utility)
        if not np.array_equal(_bits(selected), _bits(original[GROUP])):
            raise AssertionError(f"v2 G2 formula differs: {name}")
        candidate = baseline.copy()
        candidate[GROUP] = selected
        _assert_identity(candidate, baseline)
        base_path = out_dir / f"rescue_2024/{name}/baseline.parquet"
        candidate_path = out_dir / f"rescue_2024/{name}/candidate.parquet"
        detail_path = out_dir / f"rescue_2024/{name}/g2_diagnostics.parquet"
        _atomic_parquet(baseline, base_path)
        _atomic_parquet(candidate, candidate_path)
        _atomic_parquet(detail, detail_path)
        group_records: dict[str, Any] = {}
        mixed_records: dict[str, Any] = {}
        for segment_name in SEGMENT_NAMES:
            index = segments[segment_name]
            gb = _group_metrics(labels.loc[index, GROUP], baseline.loc[index, GROUP], GROUP)
            gc = _group_metrics(labels.loc[index, GROUP], candidate.loc[index, GROUP], GROUP)
            mb = _mixed_metrics(labels.loc[index], baseline.loc[index])
            mc = _mixed_metrics(labels.loc[index], candidate.loc[index])
            group_delta = gc["score"] - gb["score"]
            mixed_delta = mc["total_score"] - mb["total_score"]
            if not np.isclose(group_delta, expected["g2_group_delta_total_score"][name][segment_name], atol=2e-15, rtol=0):
                raise AssertionError(f"registered G2 evidence differs: {name}/{segment_name}")
            if not np.isclose(mixed_delta, expected["mixed_delta_total_score"][name][segment_name], atol=2e-15, rtol=0):
                raise AssertionError(f"registered mixed evidence differs: {name}/{segment_name}")
            if group_delta <= 0 or mixed_delta <= 0:
                raise AssertionError(f"2024 rescue gate fails: {name}/{segment_name}")
            group_records[segment_name] = {"baseline": gb, "candidate": gc, "delta": group_delta}
            mixed_records[segment_name] = {"baseline": mb, "candidate": mc, "delta_total_score": mixed_delta}
            if segment_name == "full":
                components = expected["mixed_full_component_deltas"][name]
                if not np.isclose(mc["one_minus_nmae"] - mb["one_minus_nmae"], components["one_minus_nmae"], atol=2e-15, rtol=0):
                    raise AssertionError("full NMAE evidence differs")
                if not np.isclose(mc["ficr"] - mb["ficr"], components["ficr"], atol=2e-15, rtol=0):
                    raise AssertionError("full FICR evidence differs")
        result[name] = {
            "g2_group": group_records,
            "mixed": mixed_records,
            "selected_rows": int(detail["gate"].sum()),
            "all_gates_passed": True,
            "outputs": [describe_file(base_path), describe_file(candidate_path), describe_file(detail_path)],
        }
    return result


def run_prescore(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    _copy_exclusive(args.config, args.out_dir / "preregister.json")
    _copy_exclusive(args.config.with_suffix(".sha256"), args.out_dir / "preregister.sha256")
    closure = _closure_record(args)
    closure_path = args.out_dir / "source_closure_before_fit_lock.json"
    _write_json(
        closure_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "locked_before_any_v5_model_fit": True,
            "locked_before_any_v5_2024_metric_reconstruction": True,
            "closure": closure,
        },
    )
    labels = _read_labels(config["input_identities"]["labels"])
    rescue_2024 = _rebuild_2024(config, args.out_dir, labels)
    result_path = args.out_dir / "rescue_2024_results.json"
    _write_json(
        result_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "posthoc_2024_rescue": True,
            "selection_unsafe": True,
            "baseline_results": rescue_2024,
        },
    )
    train_index = labels.index
    context = _read_context(config["input_identities"]["g2_train_context"], train_index)
    model = DirectIntervalProbabilityModel().fit(
        context, labels[GROUP], capacity_kwh=CAPACITY_KWH[GROUP]
    )
    if model.metadata()["parameters"] != COMMON_PARAMETERS:
        raise AssertionError("model parameters differ from source")
    model_path = args.out_dir / "final/model/kpx_group_2.joblib"
    _atomic_joblib(model, model_path)
    reloaded: DirectIntervalProbabilityModel = joblib.load(model_path)
    if reloaded.metadata() != model.metadata():
        raise AssertionError("model metadata reload differs")
    prescore_path = args.out_dir / "final_prescore_lock.json"
    _write_json(
        prescore_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "source_closure_before_fit_lock": describe_file(closure_path),
            "source_closure": closure,
            "rescue_2024_results": describe_file(result_path),
            "fresh_v5_model": describe_file(model_path),
            "model_metadata": model.metadata(),
            "model_copied_from_v4": False,
            "candidate_rule_changed_from_v4": False,
            "test_context_values_parsed_before_prescore_lock": 0,
            "recent_v4_baseline_values_parsed_before_prescore_lock": 0,
            "sample_values_parsed_before_prescore_lock": 0,
        },
    )
    print(f"source closure lock sha256: {sha256_file(closure_path)}", flush=True)
    print(f"final prescore lock sha256: {sha256_file(prescore_path)}", flush=True)


def _validate_csv(path: Path, sample: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw.startswith(b"\xef\xbb\xbf"):
        raise AssertionError("CSV BOM differs")
    text = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    if tuple(text.columns) != tuple(sample.columns) or len(text) != 8760:
        raise AssertionError("CSV schema/rows differ")
    if not text[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]].astype("string")):
        raise AssertionError("CSV ID/time differs")
    for group in TARGET_COLS:
        expected = prediction[group].reset_index(drop=True).map(lambda value: f"{float(value):.6f}").astype("string")
        if not text[group].equals(expected):
            raise AssertionError(f"CSV six-decimal values differ: {group}")
    values = text.loc[:, list(TARGET_COLS)].astype(np.float64).to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError("CSV non-finite values")
    for position, group in enumerate(TARGET_COLS):
        if values[:, position].min() < 0 or values[:, position].max() > 1.02 * CAPACITY_KWH[group] + 5e-7:
            raise AssertionError(f"CSV bounds differ: {group}")
    return {"rows": 8760, "schema": True, "sample_id_time": True, "utf8_bom": True, "six_decimals": True, "finite": True, "bounds": True}


def _supersession_exact(config: Mapping[str, Any], prediction: pd.DataFrame, detail: pd.DataFrame, csv_path: Path) -> dict[str, Any]:
    supersession = config["supersession"]
    v4_prediction = pd.read_parquet(_verify_file(supersession["quarantined_v4_predictions"])).astype(np.float64)
    v4_detail = pd.read_parquet(_verify_file(supersession["quarantined_v4_gates"]))
    if not prediction.index.equals(v4_prediction.index) or tuple(prediction.columns) != tuple(v4_prediction.columns):
        raise AssertionError("v4/v5 prediction schema differs")
    for group in TARGET_COLS:
        if not np.array_equal(_bits(prediction[group]), _bits(v4_prediction[group])):
            raise AssertionError(f"v4/v5 prediction bits differ: {group}")
    if not np.array_equal(detail["gate"].to_numpy(dtype=bool), v4_detail["gate"].to_numpy(dtype=bool)):
        raise AssertionError("v4/v5 G2 gates differ")
    v4_csv_path = _verify_file(supersession["quarantined_v4_csv"])
    v4_csv = pd.read_csv(v4_csv_path, encoding="utf-8-sig", dtype="string")
    v5_csv = pd.read_csv(csv_path, encoding="utf-8-sig", dtype="string")
    if not v4_csv.equals(v5_csv):
        raise AssertionError("v4/v5 CSV text differs")
    if v4_csv_path.read_bytes() != csv_path.read_bytes():
        raise AssertionError("v4/v5 CSV bytes differ")
    if sha256_file(csv_path) != V4_EXPECTED_CSV_SHA:
        raise AssertionError("v5 CSV hash differs from v4")
    return {
        "all_three_prediction_float64_bits_exact": True,
        "G2_gate_boolean_exact": True,
        "all_csv_columns_text_exact": True,
        "csv_bytes_exact": True,
        "shared_csv_sha256": V4_EXPECTED_CSV_SHA,
        "v4_remains_quarantined_and_submission_unauthorized": True,
    }


def _write_manifest(args: argparse.Namespace, config: Mapping[str, Any], validations: Mapping[str, Any]) -> None:
    files = sorted(
        (path for path in args.out_dir.rglob("*") if path.is_file() and path.name != "manifest.json"),
        key=lambda path: path.relative_to(args.out_dir).as_posix(),
    )
    lock = json.loads((args.out_dir / "final_prescore_lock.json").read_text(encoding="utf-8"))
    current_closure = _closure_record(args)
    _assert_closure_equal(lock["source_closure"], current_closure)
    manifest = {
        "schema_version": 1,
        "artifact_type": "direct_interval_g2_posthoc_rescue_v5_provenance_supersession",
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA,
        "risk": config["risk_classification"],
        "v4_provenance_incident": describe_file(_absolute(config["supersession"]["incident"])),
        "source_provenance": {
            "recursive_AST_local_import_closure": current_closure,
            "locked_before_fit": True,
            "recomputed_equal_before_final": True,
            "recomputed_equal_before_manifest": True,
        },
        "immutability": {
            "G2_factor": SPEC[0], "G2_margin": SPEC[1], "lookup": "linear",
            "G1_G3_float64_identity": True, "candidate_and_model_contract_unchanged_from_v4": True,
        },
        "source_closure_lock": describe_file(args.out_dir / "source_closure_before_fit_lock.json"),
        "final_prescore_lock": describe_file(args.out_dir / "final_prescore_lock.json"),
        "rescue_2024_results": describe_file(args.out_dir / "rescue_2024_results.json"),
        "supersession_exact_audit": describe_file(args.out_dir / "supersession_exact_audit.json"),
        "final_results": describe_file(args.out_dir / "final_results.json"),
        "validations": dict(validations),
        "outputs": [describe_file(path) for path in files],
        "output_count_excluding_manifest": len(files),
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
    }
    _write_json(args.out_dir / "manifest.json", manifest)


def run_final(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    prescore_path = args.out_dir / "final_prescore_lock.json"
    if not prescore_path.is_file():
        raise AssertionError("final prescore lock missing")
    prescore = json.loads(prescore_path.read_text(encoding="utf-8"))
    current_closure = _closure_record(args)
    _assert_closure_equal(prescore["source_closure"], current_closure)
    test_index = _year_index(2025)
    context = _read_context(config["input_identities"]["g2_test_context"], test_index)
    baseline_path = _verify_file(config["input_identities"]["recent_v4_baseline"])
    baseline = pd.read_parquet(baseline_path).astype(np.float64)
    baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
    if not baseline.index.equals(test_index) or tuple(baseline.columns) != TARGET_COLS:
        raise AssertionError("baseline schema/index differs")
    sample_path = _verify_file(config["input_identities"]["sample"])
    sample = pd.read_csv(sample_path, encoding="utf-8-sig", dtype={"forecast_id": "string", "forecast_kst_dtm": "string"})
    if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS) or len(sample) != 8760:
        raise AssertionError("sample schema/rows differ")
    if not pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"]), name="forecast_kst_dtm").equals(test_index):
        raise AssertionError("sample time differs")
    model_path = args.out_dir / "final/model/kpx_group_2.joblib"
    model: DirectIntervalProbabilityModel = joblib.load(model_path)
    arrays = _surface_arrays(model, context)
    surface_path = args.out_dir / "final/surface/kpx_group_2.npz"
    _atomic_npz(surface_path, arrays)
    with np.load(surface_path) as saved:
        _assert_arrays(arrays, {key: saved[key] for key in saved.files})
    reloaded: DirectIntervalProbabilityModel = joblib.load(model_path)
    _assert_arrays(arrays, _surface_arrays(reloaded, context))
    if not np.all(arrays["p6"] <= arrays["p8"]):
        raise AssertionError("p6<=p8 invariant differs")
    selected, detail = apply_rule(baseline[GROUP], arrays["utility"])
    candidate = baseline.copy()
    candidate[GROUP] = selected
    _assert_identity(candidate, baseline)
    detail_path = args.out_dir / "final/g2_diagnostics.parquet"
    prediction_path = args.out_dir / "final/predictions.parquet"
    _atomic_parquet(detail, detail_path)
    _atomic_parquet(candidate, prediction_path)
    prediction = pd.read_parquet(prediction_path).astype(np.float64)
    _assert_identity(prediction, baseline)
    for group in TARGET_COLS:
        if not np.array_equal(_bits(prediction[group]), _bits(candidate[group])):
            raise AssertionError(f"prediction parquet differs: {group}")
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = prediction[group].to_numpy(dtype=np.float64)
    csv_path = args.out_dir / str(config["output_contract"]["csv"])
    _atomic_csv(submission, csv_path)
    csv_validation = _validate_csv(csv_path, sample, prediction)
    supersession = _supersession_exact(config, prediction, detail, csv_path)
    exact_path = args.out_dir / "supersession_exact_audit.json"
    _write_json(
        exact_path,
        {
            "schema_version": 1, "created_utc": utc_now(), "status": "PASS",
            "comparison": supersession,
            "v4_prediction": config["supersession"]["quarantined_v4_predictions"],
            "v5_prediction": describe_file(prediction_path),
            "v4_gates": config["supersession"]["quarantined_v4_gates"],
            "v5_gates": describe_file(detail_path),
            "v4_csv": config["supersession"]["quarantined_v4_csv"],
            "v5_csv": describe_file(csv_path),
        },
    )
    _assert_closure_equal(prescore["source_closure"], _closure_record(args))
    validations = {
        "source_AST_closure_locked_before_fit_and_unchanged": True,
        "fresh_v5_fit": True,
        "model_reload_surface_bit_exact": True,
        "surface_npz_bit_exact": True,
        "p6_le_p8": True,
        "G2_formula_and_strict_gate_exact": True,
        "G2_selected_rows": int(detail["gate"].sum()),
        "G1_G3_float64_identity_in_memory_and_parquet": True,
        "prediction_rows_and_time": len(prediction) == 8760 and prediction.index.equals(test_index),
        "csv": csv_validation,
        "v4_v5_exact": supersession,
    }
    result_path = args.out_dir / "final_results.json"
    _write_json(
        result_path,
        {
            "schema_version": 1, "created_utc": utc_now(), "executed": True,
            "posthoc_2024_rescue": True, "selection_unsafe": True,
            "strict_final_isolation": False, "leaderboard_score_claim": False,
            "baseline": describe_file(baseline_path), "sample": describe_file(sample_path),
            "model": describe_file(model_path), "surface": describe_file(surface_path),
            "diagnostics": describe_file(detail_path), "predictions": describe_file(prediction_path),
            "submission": describe_file(csv_path), "validations": validations,
        },
    )
    _write_manifest(args, config, validations)
    print(f"CSV sha256: {sha256_file(csv_path)}", flush=True)
    print(f"manifest sha256: {sha256_file(args.out_dir / 'manifest.json')}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = _verify_config(args.config)
    if args.stage == "prescore":
        run_prescore(args, config)
    else:
        run_final(args, config)


if __name__ == "__main__":
    main()

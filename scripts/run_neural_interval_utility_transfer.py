"""Run the frozen posthoc neural interval-utility transfer candidate."""

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

from scripts import run_raw_grid_wind_lgb as raw_protocol  # noqa: E402
from src.direct_interval_probability import (  # noqa: E402
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
from src.raw_spatiotemporal_attention import RawSpatiotemporalAttentionRegressor  # noqa: E402


CONFIG_SHA = "8cc4f547662e4550eb50860f6b2d987445e57f161f5424e1ac1426c4984fbcc1"
WEIGHT = np.float64(0.05)
MARGIN = np.float64(0.01)
GROUPS = ("kpx_group_1", "kpx_group_2")
IDENTITY = "kpx_group_3"
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
YEAR_2024 = raw_protocol.YEAR_2024
YEAR_2025 = raw_protocol.YEAR_2025
PRE2024 = raw_protocol.PRE2024
TRAIN_ALL = raw_protocol.TRAIN_ALL


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prescore", "score-final"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/neural_interval_utility_transfer_preregister_v2.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/postgate/neural_interval_utility_transfer_v2"))
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


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
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
        frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.6f", lineterminator="\n")
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


def _verify_config(path: Path, *, include_final: bool) -> dict[str, Any]:
    if sha256_file(path) != CONFIG_SHA:
        raise AssertionError("config hash differs")
    if path.with_suffix(".sha256").read_text(encoding="utf-8") != f"{CONFIG_SHA}  {path.name}\n":
        raise AssertionError("config sidecar differs")
    config = json.loads(path.read_text(encoding="utf-8"))
    rule = config["immutable_single_rule"]
    if float(rule["blend_weight"]) != float(WEIGHT) or float(rule["utility_advantage_margin"]) != float(MARGIN):
        raise AssertionError("fixed rule differs")
    if rule["groups_with_rule"] != list(GROUPS) or rule["identity_groups"] != [IDENTITY]:
        raise AssertionError("fixed groups differ")
    for spec in config["lineage"].values():
        _verify(spec)
    for spec in config["input_identities"].values():
        _verify(spec)
    if include_final:
        final = config["final_if_and_only_if_dual_stage2_passes"]
        _verify(final["final_baseline"])
        for section in ("train_context", "test_context"):
            for spec in final[section].values():
                _verify(spec)
        for name in ("ldaps_test", "gfs_test", "sample"):
            _verify(final[name])
    return config


def _module_files(module: str) -> set[Path]:
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
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    hits = _module_files(alias.name)
                    if hits:
                        pending.extend(hits)
                    elif (PROJECT_DIR / alias.name.split(".")[0]).is_file():
                        raise AssertionError(f"unresolved local import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                hits = _module_files(base)
                alias_hits: set[Path] = set()
                for alias in node.names:
                    if alias.name != "*":
                        alias_hits.update(_module_files(f"{base}.{alias.name}"))
                if hits:
                    pending.extend(hits)
                if alias_hits:
                    pending.extend(alias_hits)
                elif base and (PROJECT_DIR / base.split(".")[0]).is_file():
                    raise AssertionError(f"unresolved local from-import: {base}")
    return tuple(sorted(seen))


def _closure(args: argparse.Namespace) -> dict[str, Any]:
    files = resolve_ast_closure(Path(__file__))
    return {
        "resolver": "recursive Python AST local imports plus package initializers",
        "resolved_relative_paths": [path.relative_to(PROJECT_DIR).as_posix() for path in files],
        "resolved_files": [describe_file(path) for path in files],
        "resolved_file_count": len(files),
        "unresolved_local_imports": [],
        "test": describe_file(PROJECT_DIR / "tests/test_neural_interval_utility_transfer.py"),
        "config": describe_file(args.config.resolve()),
        "sidecar": describe_file(args.config.with_suffix(".sha256").resolve()),
    }


def _spec_snapshot(specs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, spec in specs.items():
        path = _verify(spec)
        records[name] = {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return records


def _stage2_upstream_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    specs = {**config["lineage"], **config["input_identities"]}
    return _spec_snapshot(specs)


def _final_upstream_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    final = config["final_if_and_only_if_dual_stage2_passes"]
    specs: dict[str, Mapping[str, Any]] = {"final_baseline": final["final_baseline"]}
    specs.update({f"train_context_{key}": value for key, value in final["train_context"].items()})
    specs.update({f"test_context_{key}": value for key, value in final["test_context"].items()})
    specs.update({name: final[name] for name in ("ldaps_test", "gfs_test", "sample")})
    return _spec_snapshot(specs)


def _acquire_guard(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"pid": os.getpid(), "experiment_id": "neural_interval_utility_transfer_v2", "created_utc": utc_now()}
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"heavy guard already exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(owner, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    return owner


def _release_guard(path: Path, owner: Mapping[str, Any]) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return
    if int(current.get("pid", -1)) == int(owner["pid"]):
        path.unlink()


def _capacity_factors(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels.copy()
    for group in TARGET_COLS:
        result[group] = result[group] / CAPACITY_KWH[group]
    return result


def _read_prediction(spec: Mapping[str, Any], index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(_verify(spec)).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGET_COLS or not frame.index.equals(index):
        raise AssertionError("prediction schema/index differs")
    return frame


def interpolated_utility(surface: np.ndarray, query_cf: np.ndarray) -> np.ndarray:
    position = np.clip(np.asarray(query_cf, dtype=np.float64), 0.0, 1.02) * 100.0
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, 102)
    fraction = position - lower
    rows = np.arange(len(position))
    return (1.0 - fraction) * surface[rows, lower] + fraction * surface[rows, upper]


def compose_stage2(
    primary: pd.DataFrame,
    interaction: pd.DataFrame,
    neural_cf: pd.DataFrame,
    utilities: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    if not primary.index.equals(interaction.index) or not primary.index.equals(neural_cf.index):
        raise AssertionError("stage2 index differs")
    primary_candidate = primary.copy()
    interaction_candidate = interaction.copy()
    delta = pd.DataFrame(0.0, index=primary.index, columns=TARGET_COLS, dtype=np.float64)
    diagnostics: dict[str, pd.DataFrame] = {}
    for group in GROUPS:
        capacity = float(CAPACITY_KWH[group])
        base_cf = primary[group].to_numpy(dtype=np.float64) / capacity
        raw_cf = np.clip(neural_cf[group].to_numpy(dtype=np.float64), 0.0, 1.02)
        action_cf = base_cf + WEIGHT * (raw_cf - base_cf)
        base_u = interpolated_utility(utilities[group], base_cf)
        action_u = interpolated_utility(utilities[group], action_cf)
        advantage = action_u - base_u
        gate = advantage > MARGIN
        chosen = np.clip(action_cf, 0.0, 1.02) * capacity
        values = chosen - primary[group].to_numpy(dtype=np.float64)
        delta.loc[gate, group] = values[gate]
        primary_candidate.loc[gate, group] = chosen[gate]
        interaction_values = np.clip(interaction[group].to_numpy(dtype=np.float64) + values, 0.0, 1.02 * capacity)
        interaction_candidate.loc[gate, group] = interaction_values[gate]
        diagnostics[group] = pd.DataFrame(
            {"baseline_primary_cf": base_cf, "neural_raw_cf": raw_cf, "action_cf": action_cf, "base_utility": base_u, "action_utility": action_u, "utility_advantage": advantage, "gate": gate, "primary_delta_kwh": delta[group].to_numpy()},
            index=primary.index,
        )
    for frame, baseline in ((primary_candidate, primary), (interaction_candidate, interaction)):
        if not np.array_equal(frame[IDENTITY].to_numpy().view(np.uint64), baseline[IDENTITY].to_numpy().view(np.uint64)):
            raise AssertionError("G3 identity differs")
    return primary_candidate, interaction_candidate, delta, diagnostics


def _segments(year: int) -> dict[str, pd.DatetimeIndex]:
    def interval(start: str, end: str) -> pd.DatetimeIndex:
        return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")
    return {
        "full": interval(f"{year}-01-01 01:00", f"{year + 1}-01-01 00:00"),
        "H1": interval(f"{year}-01-01 01:00", f"{year}-07-01 00:00"),
        "H2": interval(f"{year}-07-01 01:00", f"{year + 1}-01-01 00:00"),
        "Q1": interval(f"{year}-01-01 01:00", f"{year}-04-01 00:00"),
        "Q2": interval(f"{year}-04-01 01:00", f"{year}-07-01 00:00"),
        "Q3": interval(f"{year}-07-01 01:00", f"{year}-10-01 00:00"),
        "Q4": interval(f"{year}-10-01 01:00", f"{year + 1}-01-01 00:00"),
    }


def _group_metric(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, float]:
    capacity = float(CAPACITY_KWH[group])
    y = actual.to_numpy(dtype=np.float64)
    p = prediction.to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & (y >= 0.10 * capacity)
    y, p = y[valid], p[valid]
    error = np.abs(p - y) / capacity
    n = 1.0 - float(error.mean())
    price = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    f = float(np.sum(y * price) / np.sum(y * 4.0))
    return {"total_score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f}


def _mixed_metric(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, float]:
    records = [_group_metric(actual[group], prediction[group], group) for group in TARGET_COLS]
    n = float(np.mean([record["one_minus_nmae"] for record in records]))
    f = float(np.mean([record["ficr"] for record in records]))
    return {"total_score": 0.5 * (n + f), "one_minus_nmae": n, "ficr": f}


def score_candidate(actual: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    all_positive = True
    segments = _segments(2024)
    for name in SEGMENTS:
        rows = segments[name]
        groups: dict[str, Any] = {}
        for group in GROUPS:
            before = _group_metric(actual.loc[rows, group], baseline.loc[rows, group], group)
            after = _group_metric(actual.loc[rows, group], candidate.loc[rows, group], group)
            delta = after["total_score"] - before["total_score"]
            groups[group] = {"baseline": before, "candidate": after, "delta_total_score": delta}
            all_positive = all_positive and delta > 0.0
        mixed_before = _mixed_metric(actual.loc[rows], baseline.loc[rows])
        mixed_after = _mixed_metric(actual.loc[rows], candidate.loc[rows])
        mixed_delta = mixed_after["total_score"] - mixed_before["total_score"]
        comparisons[name] = {"groups": groups, "mixed_baseline": mixed_before, "mixed_candidate": mixed_after, "mixed_delta_total_score": mixed_delta, "mixed_delta_one_minus_nmae": mixed_after["one_minus_nmae"] - mixed_before["one_minus_nmae"], "mixed_delta_ficr": mixed_after["ficr"] - mixed_before["ficr"]}
        all_positive = all_positive and mixed_delta > 0.0
    full = comparisons["full"]
    passed = bool(all_positive and full["mixed_delta_one_minus_nmae"] >= 0.0 and full["mixed_delta_ficr"] >= 0.0)
    return {"comparisons": comparisons, "all_21_score_deltas_strictly_positive": all_positive, "full_components_nonnegative": full["mixed_delta_one_minus_nmae"] >= 0.0 and full["mixed_delta_ficr"] >= 0.0, "passed": passed}


def _read_full_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm")), name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGET_COLS or not frame.index.equals(TRAIN_ALL):
        raise AssertionError("full labels differ")
    return frame.astype(np.float64)


def _prescore(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    _copy(args.config, args.out_dir / "preregister.json")
    _copy(args.config.with_suffix(".sha256"), args.out_dir / "preregister.sha256")
    closure = _closure(args)
    upstream = _stage2_upstream_snapshot(config)
    _write_json(args.out_dir / "source_and_upstream_before_fit_lock.json", {"schema_version": 1, "created_utc": utc_now(), "config_sha256": CONFIG_SHA, "new_fit_calls_before_lock": 0, "new_prediction_gate_delta_metric_values_before_lock": 0, "closure": closure, "upstream": upstream})
    guard_path = PROJECT_DIR / config["execution_integrity"]["heavy_guard"]
    owner = _acquire_guard(guard_path)
    try:
        raw_contract = json.loads((PROJECT_DIR / config["lineage"]["raw_grid_input_contract"]["path"]).read_text(encoding="utf-8"))
        features, weather_evidence = raw_protocol._read_weather_features(args.raw_dir.resolve(), raw_contract, period="train_all")
        prefix = raw_contract["physical_stage1_inputs"]["labels_stage1_score_prefix_after_lock"]
        labels, label_evidence = raw_protocol._read_label_prefix(args.raw_dir.resolve(), prefix, TARGET_COLS)
        if not labels.index.equals(PRE2024):
            raise AssertionError("pre2024 label prefix differs")
        model = RawSpatiotemporalAttentionRegressor(raw_contract["grid_catalog"]).fit(features.loc[PRE2024], _capacity_factors(labels))
        neural_cf = model.predict(features.loc[YEAR_2024])
        model_path = args.out_dir / "stage2/model/raw_neural_pre2024.joblib"
        raw_path = args.out_dir / "stage2/raw_neural_cf_2024.parquet"
        _atomic_joblib(model, model_path)
        _atomic_parquet(neural_cf, raw_path)
        loaded: RawSpatiotemporalAttentionRegressor = joblib.load(model_path)
        replay = loaded.predict(features.loc[YEAR_2024])
        if not np.array_equal(replay.to_numpy(), neural_cf.to_numpy()):
            raise AssertionError("raw neural reload prediction differs")
        primary = _read_prediction(config["input_identities"]["stage2_primary_baseline"], YEAR_2024)
        interaction = _read_prediction(config["input_identities"]["stage2_recent_v4_baseline"], YEAR_2024)
        utilities: dict[str, np.ndarray] = {}
        for group in GROUPS:
            spec = config["input_identities"][f"stage2_{'G1' if group == GROUPS[0] else 'G2'}_utility_surface"]
            with np.load(_verify(spec)) as payload:
                utilities[group] = np.asarray(payload["utility"], dtype=np.float64).copy()
            if utilities[group].shape != (len(YEAR_2024), 103):
                raise AssertionError("utility surface shape differs")
        primary_candidate, interaction_candidate, delta, diagnostics = compose_stage2(primary, interaction, neural_cf, utilities)
        paths = [model_path, raw_path]
        for name, frame in (("primary_baseline", primary), ("primary_candidate", primary_candidate), ("recent_v4_baseline", interaction), ("recent_v4_candidate", interaction_candidate), ("primary_delta", delta)):
            path = args.out_dir / f"stage2/{name}_2024.parquet"
            _atomic_parquet(frame, path)
            paths.append(path)
        for group, frame in diagnostics.items():
            path = args.out_dir / f"stage2/{group}_diagnostics.parquet"
            _atomic_parquet(frame, path)
            paths.append(path)
        record_path = args.out_dir / "stage2_prescore_record.json"
        _write_json(record_path, {"schema_version": 1, "created_utc": utc_now(), "config_sha256": CONFIG_SHA, "process_pid": os.getpid(), "weather_evidence": weather_evidence, "pre2024_label_evidence": label_evidence, "model_metadata": model.metadata(), "model_reload_prediction_bit_exact": True, "gate_rows": {group: int(diagnostics[group]["gate"].sum()) for group in GROUPS}, "G3_identity_both_baselines": True, "same_primary_delta_transferred_to_recent_v4": True, "outputs": [describe_file(path) for path in paths], "closure": closure, "upstream_before": upstream, "2024_application_label_values_parsed_for_this_candidate_before_lock": 0})
        if closure != _closure(args):
            raise AssertionError("source closure changed during prescore")
        if upstream != _stage2_upstream_snapshot(config):
            raise AssertionError("upstream changed during prescore")
        lock_path = args.out_dir / "candidate_before_2024_label_lock.json"
        _write_json(lock_path, {"schema_version": 1, "created_utc": utc_now(), "config_sha256": CONFIG_SHA, "prescore_record": describe_file(record_path), "candidate_outputs": [describe_file(path) for path in paths], "candidate_model_gate_delta_both_baselines_frozen": True, "created_before_2024_application_labels_were_reopened_for_this_run": True, "no_metric_values_computed": True})
        print(f"PID={os.getpid()} candidate_before_2024_label_lock_sha256={sha256_file(lock_path)}", flush=True)
    finally:
        _release_guard(guard_path, owner)


def _surface_arrays(model: DirectIntervalProbabilityModel, context: pd.DataFrame) -> dict[str, np.ndarray]:
    surface = model.predict_surfaces(context)
    return {"p6_raw": np.asarray(surface["p6_raw"]), "p8_raw": np.asarray(surface["p8_raw"]), "p6": np.asarray(surface["p6"]), "p8": np.asarray(surface["p8"]), "eae": np.asarray(surface["eae"]), "utility": official_action_utility(surface["p6"], surface["p8"], surface["eae"]), "repair_count": np.asarray([surface["repair_count"]], dtype=np.int64)}


def _read_context(spec: Mapping[str, Any], index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(_verify(spec))
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index):
        raise AssertionError("context index differs")
    return frame.loc[:, list(CONTEXT_COLUMNS)].astype(np.float64)


def _validate_csv(path: Path, sample: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    if not path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise AssertionError("CSV BOM differs")
    text = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    if tuple(text.columns) != tuple(sample.columns) or len(text) != 8760 or not text[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]]):
        raise AssertionError("CSV sample/schema differs")
    for group in TARGET_COLS:
        expected = prediction[group].reset_index(drop=True).map(lambda value: f"{float(value):.6f}").astype("string")
        if not text[group].equals(expected):
            raise AssertionError(f"CSV roundtrip differs: {group}")
    values = text.loc[:, list(TARGET_COLS)].astype(np.float64).to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError("CSV non-finite")
    for position, group in enumerate(TARGET_COLS):
        if values[:, position].min() < 0.0 or values[:, position].max() > 1.02 * CAPACITY_KWH[group] + 5e-7:
            raise AssertionError("CSV bounds differs")
    return {"rows": 8760, "sample_schema_time": True, "BOM": True, "six_decimals": True, "finite_bounds": True, "sha256": sha256_file(path)}


def _run_final(args: argparse.Namespace, config: Mapping[str, Any], labels: pd.DataFrame) -> dict[str, Any]:
    final = config["final_if_and_only_if_dual_stage2_passes"]
    snapshot = _final_upstream_snapshot(config)
    guard_path = PROJECT_DIR / config["execution_integrity"]["heavy_guard"]
    owner = _acquire_guard(guard_path)
    try:
        raw_contract = json.loads((PROJECT_DIR / config["lineage"]["raw_grid_input_contract"]["path"]).read_text(encoding="utf-8"))
        train_features, train_evidence = raw_protocol._read_weather_features(args.raw_dir.resolve(), raw_contract, period="train_all")
        test_features, test_evidence = raw_protocol._read_weather_features(args.raw_dir.resolve(), raw_contract, period="test")
        neural = RawSpatiotemporalAttentionRegressor(raw_contract["grid_catalog"]).fit(train_features.loc[TRAIN_ALL], _capacity_factors(labels))
        raw_cf = neural.predict(test_features.loc[YEAR_2025])
        neural_path = args.out_dir / "final/models/raw_neural_through2024.joblib"
        raw_path = args.out_dir / "final/raw_neural_cf_2025.parquet"
        _atomic_joblib(neural, neural_path)
        _atomic_parquet(raw_cf, raw_path)
        replay: RawSpatiotemporalAttentionRegressor = joblib.load(neural_path)
        if not np.array_equal(replay.predict(test_features.loc[YEAR_2025]).to_numpy(), raw_cf.to_numpy()):
            raise AssertionError("final raw neural reload differs")
        baseline = _read_prediction(final["final_baseline"], YEAR_2025)
        candidate = baseline.copy()
        paths = [neural_path, raw_path]
        diagnostics: dict[str, pd.DataFrame] = {}
        for group in GROUPS:
            train_context = _read_context(final["train_context"][group], TRAIN_ALL)
            test_context = _read_context(final["test_context"][group], YEAR_2025)
            direct = DirectIntervalProbabilityModel().fit(train_context, labels[group], capacity_kwh=CAPACITY_KWH[group])
            arrays = _surface_arrays(direct, test_context)
            direct_path = args.out_dir / f"final/models/direct_interval_{group}.joblib"
            surface_path = args.out_dir / f"final/surfaces/{group}.npz"
            _atomic_joblib(direct, direct_path)
            _atomic_npz(surface_path, arrays)
            repeated: DirectIntervalProbabilityModel = joblib.load(direct_path)
            rebuilt = _surface_arrays(repeated, test_context)
            for name in arrays:
                if not np.array_equal(arrays[name], rebuilt[name]):
                    raise AssertionError(f"final direct reload surface differs: {group}/{name}")
            capacity = float(CAPACITY_KWH[group])
            base_cf = baseline[group].to_numpy(dtype=np.float64) / capacity
            neural_values = np.clip(raw_cf[group].to_numpy(dtype=np.float64), 0.0, 1.02)
            action_cf = base_cf + WEIGHT * (neural_values - base_cf)
            base_u = interpolated_utility(arrays["utility"], base_cf)
            action_u = interpolated_utility(arrays["utility"], action_cf)
            advantage = action_u - base_u
            gate = advantage > MARGIN
            chosen = np.clip(action_cf, 0.0, 1.02) * capacity
            candidate.loc[gate, group] = chosen[gate]
            diagnostics[group] = pd.DataFrame({"baseline_cf": base_cf, "neural_raw_cf": neural_values, "action_cf": action_cf, "base_utility": base_u, "action_utility": action_u, "utility_advantage": advantage, "gate": gate, "delta_kwh": candidate[group].to_numpy() - baseline[group].to_numpy()}, index=YEAR_2025)
            diagnostic_path = args.out_dir / f"final/{group}_diagnostics.parquet"
            _atomic_parquet(diagnostics[group], diagnostic_path)
            paths.extend((direct_path, surface_path, diagnostic_path))
        if not np.array_equal(candidate[IDENTITY].to_numpy().view(np.uint64), baseline[IDENTITY].to_numpy().view(np.uint64)):
            raise AssertionError("final G3 identity differs")
        prediction_path = args.out_dir / "final/predictions.parquet"
        _atomic_parquet(candidate, prediction_path)
        paths.append(prediction_path)
        sample = pd.read_csv(_verify(final["sample"]), encoding="utf-8-sig", dtype="string")
        if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS) or len(sample) != 8760:
            raise AssertionError("sample differs")
        if not pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"]), name="forecast_kst_dtm").equals(YEAR_2025):
            raise AssertionError("sample time differs")
        submission = sample.copy()
        for group in TARGET_COLS:
            submission[group] = candidate[group].to_numpy()
        csv_path = args.out_dir / final["CSV"]
        _atomic_csv(submission, csv_path)
        paths.append(csv_path)
        validation = _validate_csv(csv_path, sample, candidate)
        if snapshot != _final_upstream_snapshot(config):
            raise AssertionError("final upstream changed")
        return {"executed": True, "train_weather_evidence": train_evidence, "test_weather_evidence": test_evidence, "model_reload_exact": True, "direct_surface_reload_exact": True, "gate_rows": {group: int(diagnostics[group]["gate"].sum()) for group in GROUPS}, "G3_bit_identity": True, "outputs": [describe_file(path) for path in paths], "submission": describe_file(csv_path), "validation": validation, "upstream_before_after_exact": True, "upstream": snapshot}
    finally:
        _release_guard(guard_path, owner)


def _score_final(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    lock_path = args.out_dir / "candidate_before_2024_label_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("candidate-before-label lock missing")
    for name in ("stage2_results.json", "manifest.json"):
        if (args.out_dir / name).exists():
            raise FileExistsError(args.out_dir / name)
    prescore = json.loads((args.out_dir / "stage2_prescore_record.json").read_text(encoding="utf-8"))
    if prescore["closure"] != _closure(args):
        raise AssertionError("source closure changed after prescore")
    for spec in prescore["outputs"]:
        _verify(spec)
    if prescore["upstream_before"] != _stage2_upstream_snapshot(config):
        raise AssertionError("stage2 upstream changed after prescore")
    labels = _read_full_labels(_verify(config["input_identities"]["train_labels"]))
    primary = pd.read_parquet(args.out_dir / "stage2/primary_baseline_2024.parquet")
    primary_candidate = pd.read_parquet(args.out_dir / "stage2/primary_candidate_2024.parquet")
    interaction = pd.read_parquet(args.out_dir / "stage2/recent_v4_baseline_2024.parquet")
    interaction_candidate = pd.read_parquet(args.out_dir / "stage2/recent_v4_candidate_2024.parquet")
    primary_result = score_candidate(labels.loc[YEAR_2024], primary, primary_candidate)
    interaction_result = score_candidate(labels.loc[YEAR_2024], interaction, interaction_candidate)
    promoted = bool(primary_result["passed"] and interaction_result["passed"])
    result = {"schema_version": 1, "created_utc": utc_now(), "config_sha256": CONFIG_SHA, "candidate_lock": describe_file(lock_path), "primary_v3": primary_result, "recent_v4_interaction": interaction_result, "dual_baseline_promoted": promoted, "no_retune_retry_rescue_or_alternative": True, "risk": config["risk_classification"]}
    result_path = args.out_dir / "stage2_results.json"
    _write_json(result_path, result)
    promotion_path = args.out_dir / "stage2_promotion_lock.json"
    _write_json(promotion_path, {"schema_version": 1, "created_utc": utc_now(), "config_sha256": CONFIG_SHA, "candidate_lock": describe_file(lock_path), "stage2_results": describe_file(result_path), "primary_passed": primary_result["passed"], "interaction_passed": interaction_result["passed"], "dual_baseline_promoted": promoted, "final_CSV_allowed": promoted, "no_retune": True})
    if promoted:
        final_result = _run_final(args, config, labels)
    else:
        final_result = {"executed": False, "dual_baseline_promoted": False, "test_weather_values_read_by_this_candidate_after_promotion": 0, "sample_values_read_by_this_candidate_after_promotion": 0, "CSV_created": False}
    final_path = args.out_dir / "final_results.json"
    _write_json(final_path, final_result)
    if prescore["closure"] != _closure(args):
        raise AssertionError("source closure changed before manifest")
    if prescore["upstream_before"] != _stage2_upstream_snapshot(config):
        raise AssertionError("stage2 upstream changed before manifest")
    files = sorted((path for path in args.out_dir.rglob("*") if path.is_file() and path.name != "manifest.json"), key=lambda path: path.relative_to(args.out_dir).as_posix())
    manifest = {"schema_version": 1, "artifact_type": "neural_interval_utility_transfer_v2", "created_utc": utc_now(), "config_sha256": CONFIG_SHA, "risk": config["risk_classification"], "public_and_scale_exclusion": config["public_and_scale_exclusion"], "immutable_rule": config["immutable_single_rule"], "source_provenance": {"closure": prescore["closure"], "locked_before_fit": True, "unchanged_before_manifest": True}, "stage2_upstream_before_after_exact": True, "stage2": result, "final": final_result, "outputs": [describe_file(path) for path in files], "output_count_excluding_manifest": len(files), "runtime": {"packages": package_versions()}, "git": git_state(PROJECT_DIR)}
    manifest_path = args.out_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(f"primary={primary_result['passed']} interaction={interaction_result['passed']} promoted={promoted}", flush=True)
    if promoted:
        print(f"CSV_sha256={final_result['submission']['sha256']}", flush=True)
    print(f"manifest_sha256={sha256_file(manifest_path)}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.config = args.config.resolve()
    args.raw_dir = args.raw_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.stage == "prescore" and args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    config = _verify_config(args.config, include_final=False)
    if args.stage == "prescore":
        _prescore(args, config)
    else:
        _score_final(args, config)


if __name__ == "__main__":
    main()

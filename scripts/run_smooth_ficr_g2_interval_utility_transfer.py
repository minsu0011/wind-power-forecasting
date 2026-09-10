"""Run the frozen G2-only smooth-FICR interval-utility transfer candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_neural_interval_utility_transfer as transfer  # noqa: E402
from scripts import run_raw_grid_wind_lgb as raw_protocol  # noqa: E402
from src.direct_interval_probability import (  # noqa: E402
    CONTEXT_COLUMNS,
    DirectIntervalProbabilityModel,
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
from src.raw_spatiotemporal_smooth_ficr import (  # noqa: E402
    RawSpatiotemporalSmoothFICRRegressor,
    candidate_frame as smooth_candidate_frame,
)


CONFIG_PATH = ROOT / "configs/smooth_ficr_g2_interval_utility_transfer_preregister_v1.json"
CONFIG_SIDECAR = CONFIG_PATH.with_suffix(".sha256")
EXPECTED_CONFIG_SHA = "173ac9566ed9a870f6cd87fbe85444d53899d77370e6181ddfc395d80a31b3dc"
DEFAULT_OUT_DIR = ROOT / "artifacts/postgate/smooth_ficr_g2_interval_utility_transfer_v1"
DEFAULT_RAW_DIR = Path(r"data/local/open")
GROUP = "kpx_group_2"
IDENTITY_GROUPS = ("kpx_group_1", "kpx_group_3")
WEIGHT = np.float64(0.025)
MARGIN = np.float64(0.01)
CANDIDATE_COLUMN = "node32_geoattn_conv48_e240_s2_sficr_w025"
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
YEAR_2023 = raw_protocol.YEAR_2023
YEAR_2024 = raw_protocol.YEAR_2024
YEAR_2025 = raw_protocol.YEAR_2025
PRE2024 = raw_protocol.PRE2024
TRAIN_ALL = raw_protocol.TRAIN_ALL


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "stage1-prescore", "stage1-score", "stage2-prescore", "stage2-score-final"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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


def _absolute(spec: Mapping[str, Any]) -> Path:
    path = Path(str(spec["path"]))
    return path if path.is_absolute() else ROOT / path


def _verify(spec: Mapping[str, Any]) -> Path:
    path = _absolute(spec)
    observed = describe_file(path)
    expected_size = spec.get("bytes", spec.get("size_bytes"))
    if expected_size is None:
        raise KeyError("descriptor has neither bytes nor size_bytes")
    if observed["size_bytes"] != int(expected_size) or observed["sha256"] != spec["sha256"]:
        raise AssertionError(f"input identity changed: {path}")
    return path


def _iter_specs(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"}.issubset(value):
            yield value
        for item in value.values():
            yield from _iter_specs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_specs(item)


def _spec_snapshot(config: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for key in keys:
        for spec in _iter_specs(config[key]):
            path = _verify(spec)
            records[str(spec["path"])] = describe_file(path)
    return dict(sorted(records.items()))


def _load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path != CONFIG_PATH.resolve():
        raise AssertionError("only the frozen canonical config is allowed")
    observed = sha256_file(path)
    sidecar = path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    if observed != EXPECTED_CONFIG_SHA or sidecar != EXPECTED_CONFIG_SHA:
        raise AssertionError("preregister identity changed")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != "smooth_ficr_g2_interval_utility_transfer_v1":
        raise AssertionError("experiment id changed")
    rule = config["immutable_single_candidate"]
    if (
        rule["active_group"] != GROUP
        or float(rule["blend_weight"]) != float(WEIGHT)
        or float(rule["utility_advantage_margin"]) != float(MARGIN)
        or tuple(rule["identity_groups"]) != IDENTITY_GROUPS
        or rule["recent_v4_independent_gate_forbidden"] is not True
    ):
        raise AssertionError("immutable candidate changed")
    return config


def _closure(args: argparse.Namespace) -> dict[str, Any]:
    files = list(transfer.resolve_ast_closure(Path(__file__).resolve()))
    explicit = [
        args.config.resolve(),
        args.config.with_suffix(".sha256").resolve(),
        ROOT / "tests/test_smooth_ficr_g2_interval_utility_transfer.py",
        ROOT / "artifacts/audits/smooth_ficr_g2_interval_utility_duplicate_census_v1.json",
    ]
    for path in explicit:
        if path not in files:
            files.append(path)
    files = sorted(files)
    return {
        "resolved_files": [describe_file(path) for path in files],
        "unresolved_local_imports": [],
        "focused_test": describe_file(ROOT / "tests/test_smooth_ficr_g2_interval_utility_transfer.py"),
    }


def _copy(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _acquire_guard(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {
        "pid": os.getpid(),
        "experiment_id": "smooth_ficr_g2_interval_utility_transfer_v1",
        "created_utc": utc_now(),
    }
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise RuntimeError(f"heavy guard is occupied: {path}") from exc
    try:
        os.write(descriptor, (json.dumps(owner, sort_keys=True) + "\n").encode("utf-8"))
    finally:
        os.close(descriptor)
    return owner


def _release_guard(path: Path, owner: Mapping[str, Any]) -> None:
    if not path.exists():
        raise AssertionError("heavy guard disappeared")
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != dict(owner):
        raise AssertionError("heavy guard ownership changed")
    path.unlink()


def interpolated_utility(surface: np.ndarray, query_cf: np.ndarray) -> np.ndarray:
    return transfer.interpolated_utility(surface, query_cf)


def compose_primary_delta(
    primary: pd.DataFrame,
    interaction: pd.DataFrame,
    raw_cf: pd.DataFrame,
    utility: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not primary.index.equals(interaction.index) or not primary.index.equals(raw_cf.index):
        raise AssertionError("composition indexes differ")
    if utility.shape != (len(primary), 103):
        raise AssertionError("utility surface shape differs")
    capacity = float(CAPACITY_KWH[GROUP])
    action = smooth_candidate_frame(
        primary[GROUP], raw_cf[GROUP], capacity_kwh=capacity
    )[CANDIDATE_COLUMN]
    base_cf = primary[GROUP].to_numpy(dtype=np.float64) / capacity
    action_cf = action.to_numpy(dtype=np.float64) / capacity
    base_u = interpolated_utility(utility, base_cf)
    action_u = interpolated_utility(utility, action_cf)
    advantage = action_u - base_u
    gate = advantage > MARGIN
    raw_delta = action.to_numpy(dtype=np.float64) - primary[GROUP].to_numpy(dtype=np.float64)
    delta_values = np.where(gate, raw_delta, 0.0)
    delta = pd.DataFrame(0.0, index=primary.index, columns=TARGET_COLS, dtype=np.float64)
    delta.loc[:, GROUP] = delta_values
    primary_candidate = primary.copy()
    primary_candidate.loc[:, GROUP] = np.clip(
        primary[GROUP].to_numpy(dtype=np.float64) + delta_values,
        0.0,
        1.02 * capacity,
    )
    interaction_candidate = interaction.copy()
    interaction_candidate.loc[:, GROUP] = np.clip(
        interaction[GROUP].to_numpy(dtype=np.float64) + delta_values,
        0.0,
        1.02 * capacity,
    )
    for candidate, baseline in (
        (primary_candidate, primary),
        (interaction_candidate, interaction),
    ):
        for identity in IDENTITY_GROUPS:
            if candidate[identity].to_numpy(dtype=np.float64).tobytes() != baseline[
                identity
            ].to_numpy(dtype=np.float64).tobytes():
                raise AssertionError(f"identity bits differ: {identity}")
    diagnostics = pd.DataFrame(
        {
            "baseline_primary_cf": base_cf,
            "smooth_ficr_raw_cf": raw_cf[GROUP].to_numpy(dtype=np.float64),
            "action_cf": action_cf,
            "base_utility": base_u,
            "action_utility": action_u,
            "utility_advantage": advantage,
            "gate": gate,
            "primary_delta_kwh": delta_values,
        },
        index=primary.index,
    )
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


def score_g2_identity_mix(
    actual: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
    *,
    year: int,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    all_group = True
    all_mixed = True
    for name, rows in _segments(year).items():
        before = transfer._group_metric(actual.loc[rows], baseline.loc[rows], GROUP)
        after = transfer._group_metric(actual.loc[rows], candidate.loc[rows], GROUP)
        deltas = {key: after[key] - before[key] for key in before}
        mixed = {key: value / 3.0 for key, value in deltas.items()}
        comparisons[name] = {
            "baseline": before,
            "candidate": after,
            "delta": deltas,
            "mixed_delta_by_exact_identity_additivity": mixed,
        }
        all_group = all_group and deltas["total_score"] > 0.0
        all_mixed = all_mixed and mixed["total_score"] > 0.0
    full = comparisons["full"]
    components = bool(
        full["delta"]["one_minus_nmae"] >= 0.0
        and full["delta"]["ficr"] >= 0.0
        and full["mixed_delta_by_exact_identity_additivity"]["one_minus_nmae"] >= 0.0
        and full["mixed_delta_by_exact_identity_additivity"]["ficr"] >= 0.0
    )
    passed = bool(all_group and all_mixed and components)
    return {
        "comparisons": comparisons,
        "G2_7_of_7_strict_positive": all_group,
        "mixed_7_of_7_strict_positive": all_mixed,
        "full_G2_and_mixed_components_nonnegative": components,
        "registered_strict_deltas": 14,
        "passed": passed,
    }


def _read_prediction(spec: Mapping[str, Any], index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(_verify(spec)).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGET_COLS or not frame.index.equals(index):
        raise AssertionError("prediction schema/index differs")
    return frame


def _read_context(spec: Mapping[str, Any], index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(_verify(spec))
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index):
        raise AssertionError("context index differs")
    return frame.loc[:, list(CONTEXT_COLUMNS)].astype(np.float64)


def _capacity_factors(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels.copy()
    for group in TARGET_COLS:
        result[group] = result[group] / CAPACITY_KWH[group]
    return result


def _surface_arrays(
    model: DirectIntervalProbabilityModel, context: pd.DataFrame
) -> dict[str, np.ndarray]:
    return transfer._surface_arrays(model, context)


def _read_utility(spec: Mapping[str, Any], expected_rows: int) -> np.ndarray:
    with np.load(_verify(spec)) as payload:
        utility = np.asarray(payload["utility"], dtype=np.float64).copy()
    if utility.shape != (expected_rows, 103) or not np.isfinite(utility).all():
        raise AssertionError("utility array differs")
    return utility


def _run_stage1_prescore(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> Path:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    _copy(args.config, args.out_dir / "preregister.json")
    _copy(args.config.with_suffix(".sha256"), args.out_dir / "preregister.sha256")
    _copy(_absolute(config["duplicate_census"]), args.out_dir / "duplicate_census.json")
    closure = _closure(args)
    upstream = _spec_snapshot(
        config,
        ("duplicate_census", "lineage", "stage1_input_identities"),
    )
    source_lock = args.out_dir / "source_and_upstream_before_stage1_value_parse_lock.json"
    _write_json(
        source_lock,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": EXPECTED_CONFIG_SHA,
            "closure": closure,
            "upstream": upstream,
            "stored_prediction_value_cells_parsed_before_lock": 0,
            "candidate_gate_metric_application_label_values_before_lock": 0,
            "fit_calls_before_lock": 0,
        },
    )

    specs = config["stage1_input_identities"]
    baseline = pd.read_parquet(_verify(specs["smooth_ficr_G2_baseline"])).astype(np.float64)
    raw = pd.read_parquet(_verify(specs["smooth_ficr_raw_cf"])).astype(np.float64)
    candidates = pd.read_parquet(_verify(specs["smooth_ficr_G2_candidates"])).astype(np.float64)
    for frame in (baseline, raw, candidates):
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(YEAR_2023):
            raise AssertionError("stored Stage1 index differs")
    if tuple(baseline.columns) != (GROUP,) or GROUP not in raw or CANDIDATE_COLUMN not in candidates:
        raise AssertionError("stored Stage1 columns differ")
    replay = smooth_candidate_frame(
        baseline[GROUP], raw[GROUP], capacity_kwh=float(CAPACITY_KWH[GROUP])
    )[CANDIDATE_COLUMN]
    stored_action = candidates[CANDIDATE_COLUMN]
    if replay.to_numpy(dtype=np.float64).tobytes() != stored_action.to_numpy(dtype=np.float64).tobytes():
        raise AssertionError("stored smooth-FICR w025 action does not replay bit-exactly")
    utility = _read_utility(specs["direct_interval_G2_surface"], len(YEAR_2023))
    capacity = float(CAPACITY_KWH[GROUP])
    base_cf = baseline[GROUP].to_numpy(dtype=np.float64) / capacity
    action_cf = stored_action.to_numpy(dtype=np.float64) / capacity
    base_u = interpolated_utility(utility, base_cf)
    action_u = interpolated_utility(utility, action_cf)
    advantage = action_u - base_u
    gate = advantage > MARGIN
    candidate_values = np.where(
        gate,
        stored_action.to_numpy(dtype=np.float64),
        baseline[GROUP].to_numpy(dtype=np.float64),
    )
    candidate = pd.DataFrame({GROUP: candidate_values}, index=YEAR_2023)
    delta = candidate - baseline
    diagnostics = pd.DataFrame(
        {
            "baseline_cf": base_cf,
            "smooth_ficr_raw_cf": raw[GROUP].to_numpy(dtype=np.float64),
            "stored_w025_action_cf": action_cf,
            "base_utility": base_u,
            "action_utility": action_u,
            "utility_advantage": advantage,
            "gate": gate,
            "delta_kwh": delta[GROUP].to_numpy(dtype=np.float64),
        },
        index=YEAR_2023,
    )
    paths: list[Path] = []
    for name, frame in (
        ("baseline_G2_2023", baseline),
        ("stored_w025_action_G2_2023", stored_action.rename(GROUP).to_frame()),
        ("candidate_G2_2023", candidate),
        ("delta_G2_2023", delta),
        ("diagnostics_G2_2023", diagnostics),
    ):
        path = args.out_dir / f"stage1/{name}.parquet"
        transfer._atomic_parquet(frame, path)
        paths.append(path)
    record = args.out_dir / "stage1_prescore_record.json"
    _write_json(
        record,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": EXPECTED_CONFIG_SHA,
            "source_lock": describe_file(source_lock),
            "closure": closure,
            "upstream": upstream,
            "stored_action_formula_reload_bit_exact": True,
            "gate_rows": int(np.count_nonzero(gate)),
            "G1_G3_identity_declared": True,
            "outputs": [describe_file(path) for path in paths],
            "application_label_value_cells_parsed_before_record": 0,
            "metric_calls_before_record": 0,
            "fit_calls": 0,
        },
    )
    lock = args.out_dir / "stage1_candidate_before_application_label_lock.json"
    _write_json(
        lock,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": EXPECTED_CONFIG_SHA,
            "prescore_record": describe_file(record),
            "candidate_outputs": [describe_file(path) for path in paths],
            "candidate_surface_action_gate_delta_frozen": True,
            "created_before_any_2023_application_label_value_parse_for_this_candidate": True,
            "metric_calls_before_lock": 0,
        },
    )
    if closure != _closure(args) or upstream != _spec_snapshot(
        config, ("duplicate_census", "lineage", "stage1_input_identities")
    ):
        raise AssertionError("source/upstream changed during Stage1 prescore")
    print(f"stage1_candidate_lock_sha256={sha256_file(lock)}", flush=True)
    return lock


def _run_stage1_score(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> dict[str, Any]:
    lock = args.out_dir / "stage1_candidate_before_application_label_lock.json"
    if not lock.is_file():
        raise FileNotFoundError(lock)
    for name in ("stage1_results.json", "stage1_promotion_lock.json"):
        if (args.out_dir / name).exists():
            raise FileExistsError(args.out_dir / name)
    lock_payload = json.loads(lock.read_text(encoding="utf-8"))
    if lock_payload["config_sha256"] != EXPECTED_CONFIG_SHA:
        raise AssertionError("Stage1 lock differs")
    record = json.loads((args.out_dir / "stage1_prescore_record.json").read_text(encoding="utf-8"))
    if record["closure"] != _closure(args):
        raise AssertionError("source closure changed after Stage1 lock")
    for spec in record["outputs"]:
        _verify(spec)
    if record["upstream"] != _spec_snapshot(
        config, ("duplicate_census", "lineage", "stage1_input_identities")
    ):
        raise AssertionError("Stage1 upstream changed after lock")
    raw_contract = json.loads(_verify(config["lineage"]["raw_grid_physical_contract"]).read_text(encoding="utf-8"))
    labels, evidence = raw_protocol._read_label_prefix(
        args.raw_dir.resolve(),
        raw_contract["physical_stage1_inputs"]["labels_stage1_score_prefix_after_lock"],
        (GROUP,),
    )
    if not labels.index.equals(PRE2024):
        raise AssertionError("bounded Stage1 label prefix differs")
    access = args.out_dir / "stage1_application_label_access_after_candidate_lock.json"
    _write_json(
        access,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "candidate_lock": describe_file(lock),
            "physical_read_evidence": evidence,
            "materialized_application_group": GROUP,
            "metric_calls_before_access_record": 0,
        },
    )
    baseline = pd.read_parquet(args.out_dir / "stage1/baseline_G2_2023.parquet")[GROUP]
    candidate = pd.read_parquet(args.out_dir / "stage1/candidate_G2_2023.parquet")[GROUP]
    result = score_g2_identity_mix(
        labels.loc[YEAR_2023, GROUP], baseline, candidate, year=2023
    )
    payload = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "config_sha256": EXPECTED_CONFIG_SHA,
        "candidate_lock": describe_file(lock),
        "label_access": describe_file(access),
        "risk": config["risk_classification"],
        "result": result,
        "stage1_promoted": result["passed"],
        "no_retune_retry_rescue_or_alternative": True,
    }
    result_path = args.out_dir / "stage1_results.json"
    _write_json(result_path, payload)
    promotion = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "config_sha256": EXPECTED_CONFIG_SHA,
        "candidate_lock": describe_file(lock),
        "stage1_results": describe_file(result_path),
        "stage1_promoted": result["passed"],
        "stage2_fit_allowed": result["passed"],
        "no_rule_change": True,
    }
    promotion_path = args.out_dir / "stage1_promotion_lock.json"
    _write_json(promotion_path, promotion)
    print(f"stage1_promoted={result['passed']}", flush=True)
    return promotion


def _run_stage2_prescore(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> Path:
    promotion = json.loads((args.out_dir / "stage1_promotion_lock.json").read_text(encoding="utf-8"))
    if promotion.get("stage2_fit_allowed") is not True:
        raise RuntimeError("Stage2 is not unlocked")
    lock = args.out_dir / "candidate_before_2024_application_label_lock.json"
    if lock.exists():
        raise FileExistsError(lock)
    closure = _closure(args)
    upstream = _spec_snapshot(
        config,
        ("duplicate_census", "lineage", "stage1_input_identities", "stage2_input_identities"),
    )
    guard_path = ROOT / config["execution_integrity"]["heavy_guard"]
    owner = _acquire_guard(guard_path)
    try:
        raw_contract = json.loads(_verify(config["lineage"]["raw_grid_physical_contract"]).read_text(encoding="utf-8"))
        features, weather_evidence = raw_protocol._read_weather_features(
            args.raw_dir.resolve(), raw_contract, period="train_all"
        )
        labels, label_evidence = raw_protocol._read_label_prefix(
            args.raw_dir.resolve(),
            raw_contract["physical_stage1_inputs"]["labels_stage1_score_prefix_after_lock"],
            TARGET_COLS,
        )
        if not labels.index.equals(PRE2024):
            raise AssertionError("PRE2024 fit label prefix differs")
        model = RawSpatiotemporalSmoothFICRRegressor(raw_contract["grid_catalog"]).fit(
            features.loc[PRE2024], _capacity_factors(labels)
        )
        raw_cf = model.predict(features.loc[YEAR_2024])
        model_path = args.out_dir / "stage2/model/smooth_ficr_pre2024.joblib"
        raw_path = args.out_dir / "stage2/smooth_ficr_raw_cf_2024.parquet"
        transfer._atomic_joblib(model, model_path)
        transfer._atomic_parquet(raw_cf, raw_path)
        loaded: RawSpatiotemporalSmoothFICRRegressor = joblib.load(model_path)
        replay = loaded.predict(features.loc[YEAR_2024])
        if replay.to_numpy().tobytes() != raw_cf.to_numpy().tobytes():
            raise AssertionError("smooth-FICR Stage2 reload prediction differs")
        primary = _read_prediction(config["stage2_input_identities"]["primary_v3"], YEAR_2024)
        interaction = _read_prediction(config["stage2_input_identities"]["recent_v4"], YEAR_2024)
        utility = _read_utility(
            config["stage2_input_identities"]["direct_interval_G2_surface"], len(YEAR_2024)
        )
        primary_candidate, interaction_candidate, delta, diagnostics = compose_primary_delta(
            primary, interaction, raw_cf, utility
        )
        paths = [model_path, raw_path]
        for name, frame in (
            ("primary_baseline_2024", primary),
            ("primary_candidate_2024", primary_candidate),
            ("recent_v4_baseline_2024", interaction),
            ("recent_v4_candidate_2024", interaction_candidate),
            ("primary_delta_2024", delta),
            ("G2_diagnostics_2024", diagnostics),
        ):
            path = args.out_dir / f"stage2/{name}.parquet"
            transfer._atomic_parquet(frame, path)
            paths.append(path)
        record = args.out_dir / "stage2_prescore_record.json"
        _write_json(
            record,
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "config_sha256": EXPECTED_CONFIG_SHA,
                "process_pid": os.getpid(),
                "weather_evidence": weather_evidence,
                "pre2024_fit_label_evidence": label_evidence,
                "model_metadata": model.metadata(),
                "model_reload_prediction_bit_exact": True,
                "gate_rows": int(diagnostics["gate"].sum()),
                "primary_delta_created_once": True,
                "same_primary_delta_transferred_to_recent_v4": True,
                "recent_v4_independent_gate_calls": 0,
                "G1_G3_identity_both_baselines": True,
                "outputs": [describe_file(path) for path in paths],
                "closure": closure,
                "upstream_before": upstream,
                "2024_application_label_values_parsed_before_record": 0,
                "metric_calls_before_record": 0,
            },
        )
        _write_json(
            lock,
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "config_sha256": EXPECTED_CONFIG_SHA,
                "stage1_promotion": describe_file(args.out_dir / "stage1_promotion_lock.json"),
                "prescore_record": describe_file(record),
                "candidate_outputs": [describe_file(path) for path in paths],
                "model_surface_primary_gate_delta_and_both_candidates_frozen": True,
                "created_before_2024_application_label_values_were_parsed_for_this_candidate": True,
                "no_metric_values_computed": True,
            },
        )
        if closure != _closure(args) or upstream != _spec_snapshot(
            config,
            ("duplicate_census", "lineage", "stage1_input_identities", "stage2_input_identities"),
        ):
            raise AssertionError("source/upstream changed during Stage2 prescore")
        print(
            f"PID={os.getpid()} candidate_before_2024_label_lock_sha256={sha256_file(lock)}",
            flush=True,
        )
        return lock
    finally:
        _release_guard(guard_path, owner)


def _validate_csv(
    path: Path, sample: pd.DataFrame, prediction: pd.DataFrame
) -> dict[str, Any]:
    return transfer._validate_csv(path, sample, prediction)


def _run_final(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    labels: pd.DataFrame,
) -> dict[str, Any]:
    final = config["full_inputs_after_dual_stage2_promotion_only"]
    snapshot = _spec_snapshot(config, ("full_inputs_after_dual_stage2_promotion_only",))
    guard_path = ROOT / config["execution_integrity"]["heavy_guard"]
    owner = _acquire_guard(guard_path)
    try:
        raw_contract = json.loads(_verify(config["lineage"]["raw_grid_physical_contract"]).read_text(encoding="utf-8"))
        train_features, train_evidence = raw_protocol._read_weather_features(
            args.raw_dir.resolve(), raw_contract, period="train_all"
        )
        test_features, test_evidence = raw_protocol._read_weather_features(
            args.raw_dir.resolve(), raw_contract, period="test"
        )
        neural = RawSpatiotemporalSmoothFICRRegressor(raw_contract["grid_catalog"]).fit(
            train_features.loc[TRAIN_ALL], _capacity_factors(labels)
        )
        raw_cf = neural.predict(test_features.loc[YEAR_2025])
        neural_path = args.out_dir / "final/models/smooth_ficr_through2024.joblib"
        raw_path = args.out_dir / "final/smooth_ficr_raw_cf_2025.parquet"
        transfer._atomic_joblib(neural, neural_path)
        transfer._atomic_parquet(raw_cf, raw_path)
        repeated: RawSpatiotemporalSmoothFICRRegressor = joblib.load(neural_path)
        if repeated.predict(test_features.loc[YEAR_2025]).to_numpy().tobytes() != raw_cf.to_numpy().tobytes():
            raise AssertionError("final smooth-FICR reload prediction differs")
        train_context = _read_context(final["G2_train_context"], TRAIN_ALL)
        test_context = _read_context(final["G2_test_context"], YEAR_2025)
        direct = DirectIntervalProbabilityModel().fit(
            train_context, labels[GROUP], capacity_kwh=CAPACITY_KWH[GROUP]
        )
        arrays = _surface_arrays(direct, test_context)
        direct_path = args.out_dir / "final/models/direct_interval_G2.joblib"
        surface_path = args.out_dir / "final/surfaces/G2.npz"
        transfer._atomic_joblib(direct, direct_path)
        transfer._atomic_npz(surface_path, arrays)
        direct_replay: DirectIntervalProbabilityModel = joblib.load(direct_path)
        rebuilt = _surface_arrays(direct_replay, test_context)
        for name in arrays:
            if arrays[name].tobytes() != rebuilt[name].tobytes():
                raise AssertionError(f"final direct surface reload differs: {name}")
        primary = _read_prediction(final["final_primary_v3"], YEAR_2025)
        recent = _read_prediction(final["final_recent_v4"], YEAR_2025)
        primary_candidate, candidate, delta, diagnostics = compose_primary_delta(
            primary, recent, raw_cf, arrays["utility"]
        )
        paths = [neural_path, raw_path, direct_path, surface_path]
        for name, frame in (
            ("primary_baseline_2025", primary),
            ("primary_candidate_2025", primary_candidate),
            ("recent_v4_baseline_2025", recent),
            ("predictions", candidate),
            ("primary_delta_2025", delta),
            ("G2_diagnostics_2025", diagnostics),
        ):
            path = args.out_dir / f"final/{name}.parquet"
            transfer._atomic_parquet(frame, path)
            paths.append(path)
        prescore_lock = args.out_dir / "final_candidate_before_sample_lock.json"
        _write_json(
            prescore_lock,
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "config_sha256": EXPECTED_CONFIG_SHA,
                "outputs": [describe_file(path) for path in paths],
                "primary_gate_delta_created_once": True,
                "same_primary_delta_transferred_to_recent_v4": True,
                "recent_v4_independent_gate_calls": 0,
                "G1_G3_recent_v4_bit_identity": True,
                "created_before_sample_values_for_this_candidate": True,
            },
        )
        sample = pd.read_csv(_verify(final["sample"]), encoding="utf-8-sig", dtype="string")
        if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS) or len(sample) != 8760:
            raise AssertionError("sample schema differs")
        sample_index = pd.DatetimeIndex(
            pd.to_datetime(sample["forecast_kst_dtm"]), name="forecast_kst_dtm"
        )
        if not sample_index.equals(YEAR_2025):
            raise AssertionError("sample timestamp differs")
        submission = sample.copy()
        for group in TARGET_COLS:
            submission[group] = candidate[group].to_numpy(dtype=np.float64)
        csv_path = args.out_dir / config["final_if_and_only_if_dual_stage2_passes"]["CSV"]
        transfer._atomic_csv(submission, csv_path)
        paths.append(csv_path)
        if snapshot != _spec_snapshot(config, ("full_inputs_after_dual_stage2_promotion_only",)):
            raise AssertionError("final upstream changed")
        return {
            "executed": True,
            "train_weather_evidence": train_evidence,
            "test_weather_evidence": test_evidence,
            "neural_model_reload_bit_exact": True,
            "direct_surface_reload_bit_exact": True,
            "gate_rows": int(diagnostics["gate"].sum()),
            "same_primary_delta_transferred_to_recent_v4": True,
            "recent_v4_independent_gate_calls": 0,
            "G1_G3_bit_identity": True,
            "final_candidate_lock": describe_file(prescore_lock),
            "outputs": [describe_file(path) for path in paths],
            "submission": describe_file(csv_path),
            "validation": _validate_csv(csv_path, sample, candidate),
            "upstream_before_after_exact": True,
        }
    finally:
        _release_guard(guard_path, owner)


def _run_stage2_score_final(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> dict[str, Any]:
    lock = args.out_dir / "candidate_before_2024_application_label_lock.json"
    if not lock.is_file():
        raise FileNotFoundError(lock)
    for name in ("stage2_results.json", "stage2_promotion_lock.json", "final_results.json", "manifest.json"):
        if (args.out_dir / name).exists():
            raise FileExistsError(args.out_dir / name)
    record = json.loads((args.out_dir / "stage2_prescore_record.json").read_text(encoding="utf-8"))
    if record["closure"] != _closure(args):
        raise AssertionError("source closure changed after Stage2 candidate lock")
    for spec in record["outputs"]:
        _verify(spec)
    if record["upstream_before"] != _spec_snapshot(
        config,
        ("duplicate_census", "lineage", "stage1_input_identities", "stage2_input_identities"),
    ):
        raise AssertionError("Stage2 upstream changed after lock")
    raw_contract = json.loads(_verify(config["lineage"]["raw_grid_physical_contract"]).read_text(encoding="utf-8"))
    labels, label_evidence = raw_protocol._read_full_labels(
        args.raw_dir.resolve(), raw_contract, TARGET_COLS
    )
    if not labels.index.equals(TRAIN_ALL):
        raise AssertionError("full train labels differ")
    access = args.out_dir / "stage2_2024_application_label_access_after_candidate_lock.json"
    _write_json(
        access,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "candidate_lock": describe_file(lock),
            "physical_read_evidence": label_evidence,
            "metric_calls_before_access_record": 0,
        },
    )
    primary = pd.read_parquet(args.out_dir / "stage2/primary_baseline_2024.parquet")
    primary_candidate = pd.read_parquet(args.out_dir / "stage2/primary_candidate_2024.parquet")
    recent = pd.read_parquet(args.out_dir / "stage2/recent_v4_baseline_2024.parquet")
    recent_candidate = pd.read_parquet(args.out_dir / "stage2/recent_v4_candidate_2024.parquet")
    primary_result = score_g2_identity_mix(
        labels.loc[YEAR_2024, GROUP], primary[GROUP], primary_candidate[GROUP], year=2024
    )
    recent_result = score_g2_identity_mix(
        labels.loc[YEAR_2024, GROUP], recent[GROUP], recent_candidate[GROUP], year=2024
    )
    promoted = bool(primary_result["passed"] and recent_result["passed"])
    payload = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "config_sha256": EXPECTED_CONFIG_SHA,
        "candidate_lock": describe_file(lock),
        "label_access": describe_file(access),
        "primary_v3": primary_result,
        "recent_v4_same_primary_delta_interaction": recent_result,
        "dual_baseline_promoted": promoted,
        "recent_v4_independent_gate_calls": 0,
        "no_retune_retry_rescue_or_alternative": True,
        "risk": config["risk_classification"],
    }
    result_path = args.out_dir / "stage2_results.json"
    _write_json(result_path, payload)
    promotion_path = args.out_dir / "stage2_promotion_lock.json"
    _write_json(
        promotion_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": EXPECTED_CONFIG_SHA,
            "candidate_lock": describe_file(lock),
            "stage2_results": describe_file(result_path),
            "primary_passed": primary_result["passed"],
            "recent_v4_interaction_passed": recent_result["passed"],
            "dual_baseline_promoted": promoted,
            "final_CSV_allowed": promoted,
            "no_retune": True,
        },
    )
    if promoted:
        final_result = _run_final(args, config, labels)
    else:
        final_result = {
            "executed": False,
            "dual_baseline_promoted": False,
            "test_weather_value_cells_read_for_this_candidate": 0,
            "sample_value_cells_read_for_this_candidate": 0,
            "CSV_created": False,
        }
    _write_json(args.out_dir / "final_results.json", final_result)
    print(
        f"primary={primary_result['passed']} recent={recent_result['passed']} promoted={promoted}",
        flush=True,
    )
    return payload


def _write_skipped_after_stage1(args: argparse.Namespace) -> None:
    payload = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "config_sha256": EXPECTED_CONFIG_SHA,
        "stage1_promoted": False,
        "stage2_executed": False,
        "PRE2024_refit_calls": 0,
        "year_2024_application_label_value_cells_read_for_this_candidate": 0,
        "year_2025_value_cells_read_for_this_candidate": 0,
        "CSV_created": False,
        "no_retune_retry_rescue_or_alternative": True,
    }
    _write_json(args.out_dir / "stage2_results.json", payload)
    _write_json(args.out_dir / "final_results.json", payload)


def _write_manifest(args: argparse.Namespace, config: Mapping[str, Any]) -> Path:
    path = args.out_dir / "manifest.json"
    if path.exists():
        raise FileExistsError(path)
    files = sorted(
        (
            item
            for item in args.out_dir.rglob("*")
            if item.is_file() and item != path and ".tmp-" not in item.name
        ),
        key=lambda item: item.relative_to(args.out_dir).as_posix(),
    )
    stage1 = json.loads((args.out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    stage2 = json.loads((args.out_dir / "stage2_results.json").read_text(encoding="utf-8"))
    final = json.loads((args.out_dir / "final_results.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "artifact_type": "smooth_ficr_g2_interval_utility_transfer_v1",
        "created_utc": utc_now(),
        "config_sha256": EXPECTED_CONFIG_SHA,
        "risk": config["risk_classification"],
        "public_and_scale_exclusion": config["public_and_scale_exclusion"],
        "immutable_candidate": config["immutable_single_candidate"],
        "source_provenance": {
            "closure": _closure(args),
            "locked_before_stored_prediction_value_parse": True,
        },
        "stage1": stage1,
        "stage2": stage2,
        "final": final,
        "outputs": [describe_file(item) for item in files],
        "output_count_excluding_manifest": len(files),
        "runtime": {"packages": package_versions()},
        "git": git_state(ROOT),
        "no_rule_change_retune_retry_rescue_or_alternative": True,
    }
    _write_json(path, payload)
    sidecar = args.out_dir / "manifest.sha256"
    sidecar.write_text(sha256_file(path) + "\n", encoding="ascii")
    print(f"manifest_sha256={sha256_file(path)}", flush=True)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.config = args.config.resolve()
    args.raw_dir = args.raw_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    config = _load_config(args.config)
    if args.stage in ("all", "stage1-prescore"):
        _run_stage1_prescore(args, config)
    if args.stage in ("all", "stage1-score"):
        promotion = _run_stage1_score(args, config)
        if args.stage == "all" and promotion["stage2_fit_allowed"] is not True:
            _write_skipped_after_stage1(args)
            _write_manifest(args, config)
            return
    if args.stage in ("all", "stage2-prescore"):
        _run_stage2_prescore(args, config)
    if args.stage in ("all", "stage2-score-final"):
        _run_stage2_score_final(args, config)
        _write_manifest(args, config)


if __name__ == "__main__":
    main()

"""Independent read-only audit for the promoted G1/G2 joint-NWP submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1"
MANIFEST_PATH = FINAL_ROOT / "manifest.json"
MANIFEST_SIDECAR = FINAL_ROOT / "manifest.sha256"
FINAL_AUDIT_PATH = FINAL_ROOT / "final_audit.json"
SOURCE_LOCK_PATH = FINAL_ROOT / "source_lock_before_label_read_or_final_fit.json"
LABEL_LOCK_PATH = FINAL_ROOT / "label_lock_before_final_fit.json"
CONFIG_PATH = ROOT / "configs/multi_nwp_joint_g12_final_preregister_v1.json"
CONFIG_SIDECAR = CONFIG_PATH.with_suffix(".sha256")
TEST_INDEX = pd.date_range(
    "2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm"
)
MODEL_APPLY_INDEX = pd.date_range(
    "2025-01-01 01:00", "2025-12-31 23:00", freq="h", name="forecast_kst_dtm"
)
TARGETS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
ACTIVE_GROUPS = ("kpx_group_1", "kpx_group_2")
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
SOURCE_ORDER = ("ecmwf", "icon", "gfs")
SOURCE_ID = {"ecmwf": "ecmwf_ifs025", "icon": "icon_global", "gfs": "gfs_global"}
JOINT_COLUMNS = (
    "mnwp__ecmwf_ws100",
    "mnwp__ecmwf_u100",
    "mnwp__ecmwf_v100",
    "mnwp__icon_ws100_vector_interp",
    "mnwp__icon_u100_vector_interp",
    "mnwp__icon_v100_vector_interp",
    "mnwp__gfs_ws100",
    "mnwp__gfs_u100",
    "mnwp__gfs_v100",
    "mnwp__ws100_mean",
    "mnwp__ws100_std",
    "mnwp__ws100_range",
    "mnwp__u100_mean",
    "mnwp__u100_std",
    "mnwp__u100_range",
    "mnwp__v100_mean",
    "mnwp__v100_std",
    "mnwp__v100_range",
    "mnwp__consensus_vector_speed",
    "mnwp__directional_coherence",
    "mnwp__vector_distance_ecmwf_icon",
    "mnwp__vector_distance_ecmwf_gfs",
    "mnwp__vector_distance_icon_gfs",
    "mnwp__ecmwf_minus_canonical_hub_ws",
    "mnwp__icon_minus_canonical_hub_ws",
    "mnwp__gfs_minus_canonical_hub_ws",
    "mnwp__mean_minus_canonical_hub_ws",
    "mnwp__spread_to_mean_ws_ratio",
    "mnwp__vertical_shear_mean",
    "mnwp__vertical_shear_abs_disagreement",
)
PREVIOUS_RUNS_DOC = "https://open-meteo.com/en/docs/previous-runs-api"
LICENSE_URL = "https://open-meteo.com/en/license"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def canonical_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_record(record: Mapping[str, Any]) -> dict[str, Any]:
    path = canonical_path(str(record["path"]))
    actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if actual["bytes"] != int(record["bytes"]) or actual["sha256"] != str(record["sha256"]).lower():
        raise AssertionError(f"file record differs: {path}")
    actual["record_exact"] = True
    return actual


def walk_records(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"}.issubset(value):
            yield value
        for child in value.values():
            yield from walk_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_records(child)


def unique_verified(value: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for record in walk_records(value):
        key = (
            str(canonical_path(str(record["path"]))),
            int(record["bytes"]),
            str(record["sha256"]).lower(),
        )
        if key not in seen:
            seen.add(key)
            output.append(verify_record(record))
    return output


def tree_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        result[path.relative_to(root).as_posix()] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256(path),
        }
    return result


def snapshot_sha(snapshot: Mapping[str, Any]) -> str:
    raw = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_source(path: Path, group: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    local = frame.loc[frame["group"] == group].drop(columns="group").set_index("time")
    local.index = pd.DatetimeIndex(local.index, name="forecast_kst_dtm")
    local = local.loc[index].astype(np.float64)
    if not local.index.equals(index) or not np.isfinite(local.to_numpy()).all():
        raise AssertionError(f"source coverage differs: {path}/{group}")
    return local


def selected(frame: pd.DataFrame, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    day1 = frame.index.hour.astype(int).isin(range(1, 14))
    speed = np.where(
        day1,
        frame[f"wind_speed_{height}m_previous_day1"].to_numpy(dtype=np.float64),
        frame[f"wind_speed_{height}m_previous_day2"].to_numpy(dtype=np.float64),
    )
    direction = np.where(
        day1,
        frame[f"wind_direction_{height}m_previous_day1"].to_numpy(dtype=np.float64),
        frame[f"wind_direction_{height}m_previous_day2"].to_numpy(dtype=np.float64),
    )
    radians = np.deg2rad(direction)
    return speed, -speed * np.sin(radians), -speed * np.cos(radians)


def build_joint_features(
    sources: Mapping[str, pd.DataFrame], canonical_hub_ws: pd.Series
) -> pd.DataFrame:
    if tuple(sources) != SOURCE_ORDER:
        raise AssertionError("source order differs")
    e_ws, e_u, e_v = selected(sources["ecmwf"], 100)
    i80_ws, i80_u, i80_v = selected(sources["icon"], 80)
    i120_ws, i120_u, i120_v = selected(sources["icon"], 120)
    i_u = 0.5 * (i80_u + i120_u)
    i_v = 0.5 * (i80_v + i120_v)
    i_ws = np.hypot(i_u, i_v)
    g80_ws, _, _ = selected(sources["gfs"], 80)
    g_ws, g_u, g_v = selected(sources["gfs"], 100)
    ws = np.column_stack((e_ws, i_ws, g_ws))
    u = np.column_stack((e_u, i_u, g_u))
    v = np.column_stack((e_v, i_v, g_v))
    ws_mean = ws.mean(axis=1)
    ws_std = ws.std(axis=1, ddof=0)
    u_mean = u.mean(axis=1)
    v_mean = v.mean(axis=1)
    vector_speed = np.hypot(u_mean, v_mean)
    hub = canonical_hub_ws.to_numpy(dtype=np.float64)
    icon_shear = i120_ws - i80_ws
    gfs_shear = g_ws - g80_ws
    values = {
        "mnwp__ecmwf_ws100": e_ws,
        "mnwp__ecmwf_u100": e_u,
        "mnwp__ecmwf_v100": e_v,
        "mnwp__icon_ws100_vector_interp": i_ws,
        "mnwp__icon_u100_vector_interp": i_u,
        "mnwp__icon_v100_vector_interp": i_v,
        "mnwp__gfs_ws100": g_ws,
        "mnwp__gfs_u100": g_u,
        "mnwp__gfs_v100": g_v,
        "mnwp__ws100_mean": ws_mean,
        "mnwp__ws100_std": ws_std,
        "mnwp__ws100_range": np.ptp(ws, axis=1),
        "mnwp__u100_mean": u_mean,
        "mnwp__u100_std": u.std(axis=1, ddof=0),
        "mnwp__u100_range": np.ptp(u, axis=1),
        "mnwp__v100_mean": v_mean,
        "mnwp__v100_std": v.std(axis=1, ddof=0),
        "mnwp__v100_range": np.ptp(v, axis=1),
        "mnwp__consensus_vector_speed": vector_speed,
        "mnwp__directional_coherence": vector_speed / np.maximum(ws_mean, 0.05),
        "mnwp__vector_distance_ecmwf_icon": np.hypot(e_u - i_u, e_v - i_v),
        "mnwp__vector_distance_ecmwf_gfs": np.hypot(e_u - g_u, e_v - g_v),
        "mnwp__vector_distance_icon_gfs": np.hypot(i_u - g_u, i_v - g_v),
        "mnwp__ecmwf_minus_canonical_hub_ws": e_ws - hub,
        "mnwp__icon_minus_canonical_hub_ws": i_ws - hub,
        "mnwp__gfs_minus_canonical_hub_ws": g_ws - hub,
        "mnwp__mean_minus_canonical_hub_ws": ws_mean - hub,
        "mnwp__spread_to_mean_ws_ratio": ws_std / np.maximum(ws_mean, 0.05),
        "mnwp__vertical_shear_mean": 0.5 * (icon_shear + gfs_shear),
        "mnwp__vertical_shear_abs_disagreement": np.abs(icon_shear - gfs_shear),
    }
    result = pd.DataFrame(index=canonical_hub_ws.index)
    for name in JOINT_COLUMNS:
        result[name] = values[name]
    result = result.astype(np.float32)
    if tuple(result.columns) != JOINT_COLUMNS or not np.isfinite(result.to_numpy()).all():
        raise AssertionError("joint feature block differs")
    return result


def validate_request(request: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    parsed = urlparse(str(request["url"]))
    query = parse_qs(parsed.query)
    variables = query.get("hourly", [""])[0].split(",")
    if (
        parsed.scheme != "https"
        or parsed.hostname != "previous-runs-api.open-meteo.com"
        or query.get("models") != [source_id]
        or query.get("timezone") != ["Asia/Seoul"]
        or query.get("cell_selection") != ["nearest"]
    ):
        raise AssertionError(f"request provenance differs: {source_id}/{request.get('group')}")
    if any(
        not (name.endswith("_previous_day1") or name.endswith("_previous_day2"))
        for name in variables
    ):
        raise AssertionError("non-previous-run request variable found")
    return {
        "source": source_id,
        "group": request["group"],
        "request_rows": int(request["request_rows"]),
        "retained_target_rows": int(request["retained_target_rows"]),
        "previous_day1_and_day2_only": True,
        "raw_response": verify_record(request["raw_response"]),
    }


def locked_frame(record: Mapping[str, Any]) -> pd.DataFrame:
    verified = verify_record(record)
    frame = pd.read_parquet(verified["path"])
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(TEST_INDEX) or tuple(frame.columns) != TARGETS:
        raise AssertionError(f"locked output frame differs: {verified['path']}")
    return frame.astype(np.float64)


def main(output: Path) -> None:
    output = output.resolve()
    if FINAL_ROOT.resolve() in output.parents:
        raise ValueError("audit must be written outside immutable candidate")
    if output.exists():
        raise FileExistsError(output)

    before = tree_snapshot(FINAL_ROOT)
    manifest = load_json(MANIFEST_PATH)
    final_audit = load_json(FINAL_AUDIT_PATH)
    source_lock = load_json(SOURCE_LOCK_PATH)
    label_lock = load_json(LABEL_LOCK_PATH)
    config = load_json(CONFIG_PATH)

    expected_config_sha = CONFIG_SIDECAR.read_text(encoding="ascii").split()[0].lower()
    if sha256(CONFIG_PATH) != expected_config_sha or verify_record(source_lock["preregister"])[
        "sha256"
    ] != expected_config_sha:
        raise AssertionError("preregister lock differs")
    if config["selection_status"] != "posthoc_selection_unsafe":
        raise AssertionError("selection status differs")
    for relative, record in config["bound_files"].items():
        path = ROOT / relative
        if path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
            raise AssertionError(f"bound preregister file differs: {relative}")
    verify_record(label_lock["source_lock"])

    manifest_records = unique_verified(manifest)
    final_audit_records = unique_verified(final_audit)
    source_lock_records = unique_verified(source_lock)
    source_manifest_record = verify_record(source_lock["source_2025_manifest"])
    source_manifest = load_json(Path(source_manifest_record["path"]))
    if source_manifest["official_documentation"] != PREVIOUS_RUNS_DOC:
        raise AssertionError("Previous Runs documentation binding differs")
    if source_manifest["cutoff_semantics"]["post_cutoff_forecast_or_observation_used"]:
        raise AssertionError("source manifest declares post-cutoff data")

    requests: list[dict[str, Any]] = []
    test_paths: dict[str, Path] = {}
    for source_key in SOURCE_ORDER:
        source_id = SOURCE_ID[source_key]
        entry = source_manifest["sources"][source_id]
        requests.extend(validate_request(item, source_id) for item in entry["requests"])
        if {item["group"] for item in entry["requests"]} != set(TARGETS):
            raise AssertionError(f"request group coverage differs: {source_id}")
        normalized = verify_record(entry["normalized"])
        test_paths[source_key] = Path(normalized["path"])
    if len(requests) != 9:
        raise AssertionError("expected nine archived API requests")

    locked_2025 = {item["sha256"] for item in source_lock["source_2025"]}
    if {sha256(path) for path in test_paths.values()} != locked_2025:
        raise AssertionError("source-lock/manifest normalized archives differ")
    terminal_coverage: dict[str, Any] = {}
    for source_key, path in test_paths.items():
        whole = pd.read_parquet(path)
        whole["time"] = pd.to_datetime(whole["time"], errors="raise")
        terminal = whole.loc[whole["time"] == pd.Timestamp("2026-01-01 00:00")]
        values = terminal.drop(columns=["group", "time"]).to_numpy(dtype=np.float64)
        if set(terminal["group"]) != set(TARGETS) or not np.isfinite(values).all():
            raise AssertionError(f"terminal archive row incomplete: {source_key}")
        terminal_coverage[source_key] = {
            "path": str(path),
            "terminal_group_rows": len(terminal),
            "terminal_values_finite": True,
            "terminal_excluded_from_model_application": True,
        }

    train_archives: list[dict[str, Any]] = []
    for item in source_lock["source_2024"]:
        train_archives.append(verify_record(item))

    model_records = {Path(item["path"]).name: item for item in final_audit["artifacts"]["models"]}
    replay_increment = pd.DataFrame(0.0, index=TEST_INDEX, columns=TARGETS, dtype=np.float64)
    reload_audit: list[dict[str, Any]] = []
    for group in ACTIVE_GROUPS:
        cache_record = next(
            item
            for item in source_lock["canonical_weather"]
            if Path(item["path"]).name == f"{group}_weather_test.parquet"
        )
        verified_cache = verify_record(cache_record)
        control_all = pd.read_parquet(verified_cache["path"])
        control_all.index = pd.DatetimeIndex(control_all.index, name="forecast_kst_dtm")
        control = control_all.loc[MODEL_APPLY_INDEX].astype(np.float32)
        if control.shape != (len(MODEL_APPLY_INDEX), 612):
            raise AssertionError(f"control apply schema differs: {group}")
        sources = {
            name: load_source(test_paths[name], group, MODEL_APPLY_INDEX) for name in SOURCE_ORDER
        }
        joint = build_joint_features(sources, control["cross__hub_ws_mean"])
        extended = pd.concat([control, joint], axis=1)
        predictions: dict[str, np.ndarray] = {}
        for kind, features, filename in (
            ("control", control, f"{group}__control.joblib"),
            ("joint_extended", extended, f"{group}__joint_extended.joblib"),
        ):
            verified_model = verify_record(model_records[filename])
            first_model = joblib.load(verified_model["path"])
            second_model = joblib.load(verified_model["path"])
            first_raw = np.asarray(first_model.predict(features), dtype=np.float64)
            second_raw = np.asarray(second_model.predict(features), dtype=np.float64)
            if not np.array_equal(first_raw, second_raw):
                raise AssertionError(f"double reload prediction differs: {filename}")
            clipped = np.clip(first_raw, 0.0, 1.02)
            predictions[kind] = clipped
            reload_audit.append(
                {
                    "group": group,
                    "kind": kind,
                    "model": verified_model,
                    "rows": len(features),
                    "features": features.shape[1],
                    "two_independent_joblib_loads": True,
                    "raw_prediction_float64_bit_exact": True,
                    "raw_prediction_sha256": array_sha256(first_raw),
                    "clipped_prediction_sha256": array_sha256(clipped),
                }
            )
        replay_increment.loc[MODEL_APPLY_INDEX, group] = (
            predictions["joint_extended"] - predictions["control"]
        )
    if len(reload_audit) != 4:
        raise AssertionError("model reload coverage is not exactly four")

    base_raw_record = verify_record(source_lock["baseline"])
    base_raw = pd.read_parquet(base_raw_record["path"])
    base_raw.index = pd.DatetimeIndex(base_raw.index, name="forecast_kst_dtm")
    replay_base = base_raw.loc[TEST_INDEX, list(TARGETS)].astype(np.float64)
    for group in TARGETS:
        replay_base[group] = np.clip(
            0.97 * replay_base[group].to_numpy(), 0.0, 1.02 * CAPACITY[group]
        )
    locked_base = locked_frame(final_audit["artifacts"]["base"])
    locked_increment = locked_frame(final_audit["artifacts"]["increment"])
    locked_prediction = locked_frame(final_audit["artifacts"]["prediction"])
    if not np.array_equal(replay_base.to_numpy(), locked_base.to_numpy()):
        raise AssertionError("0.97 base replay differs")
    if not np.array_equal(replay_increment.to_numpy(), locked_increment.to_numpy()):
        raise AssertionError("paired model increment replay differs")

    replay_prediction = replay_base.copy()
    for group in ACTIVE_GROUPS:
        replay_prediction.loc[MODEL_APPLY_INDEX, group] = np.clip(
            replay_base.loc[MODEL_APPLY_INDEX, group].to_numpy()
            + 0.25
            * CAPACITY[group]
            * replay_increment.loc[MODEL_APPLY_INDEX, group].to_numpy(),
            0.0,
            1.02 * CAPACITY[group],
        )
    if not np.array_equal(replay_prediction.to_numpy(), locked_prediction.to_numpy()):
        raise AssertionError("0.25 paired-transfer prediction replay differs")
    terminal = pd.Timestamp("2026-01-01 00:00")
    if not np.array_equal(
        replay_increment.loc[terminal].to_numpy(), np.zeros(3, dtype=np.float64)
    ) or not np.array_equal(
        replay_prediction.loc[terminal].to_numpy(), replay_base.loc[terminal].to_numpy()
    ):
        raise AssertionError("terminal zero-increment/base-identity contract differs")
    if not np.array_equal(
        replay_prediction["kpx_group_3"].to_numpy(), replay_base["kpx_group_3"].to_numpy()
    ) or not np.array_equal(
        replay_increment["kpx_group_3"].to_numpy(), np.zeros(8760, dtype=np.float64)
    ):
        raise AssertionError("G3 identity contract differs")

    csv_record = verify_record(final_audit["artifacts"]["csv"])
    sample_record = verify_record(source_lock["sample"])
    csv_path = Path(csv_record["path"])
    sample = pd.read_csv(sample_record["path"], encoding="utf-8-sig", dtype="string", keep_default_na=False)
    rendered = pd.read_csv(csv_path, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    if len(rendered) != 8760 or tuple(rendered.columns) != (
        "forecast_id",
        "forecast_kst_dtm",
        *TARGETS,
    ):
        raise AssertionError("CSV schema differs")
    if not rendered.loc[:, ["forecast_id", "forecast_kst_dtm"]].equals(
        sample.loc[:, ["forecast_id", "forecast_kst_dtm"]]
    ):
        raise AssertionError("CSV identifiers differ from sample")
    pattern = re.compile(r"^-?\d+\.\d{6}$")
    for group in TARGETS:
        expected = replay_prediction[group].round(6).to_numpy(dtype=np.float64)
        observed = rendered[group].to_numpy(dtype=np.float64)
        if not np.array_equal(expected, observed):
            raise AssertionError(f"six-decimal numeric replay differs: {group}")
        if not rendered[group].map(lambda value: bool(pattern.fullmatch(value))).all():
            raise AssertionError(f"CSV is not rendered with exactly six decimals: {group}")
        if not np.isfinite(observed).all() or (observed < 0.0).any() or (
            observed > 1.02 * CAPACITY[group]
        ).any():
            raise AssertionError(f"CSV bounds differ: {group}")
    raw_csv = csv_path.read_bytes()
    if not raw_csv.startswith(b"\xef\xbb\xbf") or raw_csv.startswith(b"\xef\xbb\xbf\xef\xbb\xbf"):
        raise AssertionError("CSV does not have exactly one BOM")
    if b"\r" in raw_csv or raw_csv.count(b"\n") != 8761:
        raise AssertionError("CSV LF/line count differs")

    manifest_sidecar_value = MANIFEST_SIDECAR.read_text(encoding="ascii").split()[0].lower()
    if sha256(MANIFEST_PATH) != manifest_sidecar_value:
        raise AssertionError("manifest sidecar differs")
    declared = {
        canonical_path(item["path"]).relative_to(FINAL_ROOT.resolve()).as_posix()
        for item in manifest["outputs_excluding_manifest_and_sidecar"]
    }
    actual = set(before)
    if manifest["unlisted_files_before_manifest"] or actual != declared | {
        "manifest.json",
        "manifest.sha256",
    }:
        raise AssertionError("manifest closure differs")

    after = tree_snapshot(FINAL_ROOT)
    if before != after:
        raise AssertionError("candidate tree changed during audit")

    audit = {
        "schema_version": 1,
        "audit_id": "multi_nwp_joint_g12_independent_readonly_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": str(FINAL_ROOT.resolve()),
        "auditor_script": {
            "path": str(Path(__file__).resolve()),
            "bytes": Path(__file__).stat().st_size,
            "sha256": sha256(Path(__file__)),
        },
        "verdict": "GO_ARTIFACT_INTEGRITY_AND_CUTOFF_PASS_SELECTION_UNSAFE",
        "selection_status": "posthoc_selection_unsafe",
        "scope_note": "The audit validates reproducibility and submission integrity, not 2025 score.",
        "checks": {
            "source_archive_and_cutoff": {
                "passed": True,
                "provider": source_manifest["provider"],
                "documentation": PREVIOUS_RUNS_DOC,
                "license": {"id": "CC-BY-4.0", "url": LICENSE_URL},
                "cutoff_selector": "KST 01-13 previous_day1; KST 00 and 14-23 previous_day2",
                "documented_offsets": "previous_day1=24h before valid time; previous_day2=48h before valid time",
                "post_cutoff_forecast_or_observation_used": False,
                "api_request_count": len(requests),
                "api_requests": requests,
                "normalized_2025_archive_records": [
                    verify_record(item) for item in source_lock["source_2025"]
                ],
                "normalized_2024_archive_records": train_archives,
                "terminal_archive_coverage": terminal_coverage,
                "terminal_source_row_used_by_model": False,
                "latest_model_application_source_time": str(MODEL_APPLY_INDEX.max()),
            },
            "four_model_double_reload": {
                "passed": True,
                "model_count": len(reload_audit),
                "loads_per_model": 2,
                "records": reload_audit,
            },
            "formula_replay": {
                "passed": True,
                "base_clip_097_float64_exact": True,
                "paired_increment_extended_minus_control_float64_exact": True,
                "transfer_weight": 0.25,
                "candidate_clip_base_plus_weight_capacity_increment_float64_exact": True,
                "g3_increment_exact_zero": True,
                "g3_prediction_exact_base_identity": True,
                "terminal_timestamp": str(terminal),
                "terminal_increment_exact_zero_all_groups": True,
                "terminal_prediction_exact_base_identity_all_groups": True,
            },
            "submission_csv": {
                "passed": True,
                "file": csv_record,
                "sample": sample_record,
                "rows": 8760,
                "columns": ["forecast_id", "forecast_kst_dtm", *TARGETS],
                "identifier_and_time_exact_sample": True,
                "single_utf8_bom": True,
                "line_endings": "LF",
                "numeric_decimals": 6,
                "numeric_round6_replay_exact": True,
                "finite": True,
                "bounds": {group: [0.0, 1.02 * CAPACITY[group]] for group in TARGETS},
                "all_values_within_bounds": True,
            },
            "manifest_closure_and_nonmutation": {
                "passed": True,
                "manifest_sha256": sha256(MANIFEST_PATH),
                "manifest_sidecar_exact": True,
                "manifest_record_count": len(manifest_records),
                "final_audit_record_count": len(final_audit_records),
                "source_lock_record_count": len(source_lock_records),
                "candidate_file_count": len(actual),
                "unlisted_candidate_files": [],
                "candidate_tree_snapshot_sha256_before": snapshot_sha(before),
                "candidate_tree_snapshot_sha256_after": snapshot_sha(after),
                "candidate_tree_byte_hash_mtime_exact_nonmutation": True,
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "verdict": audit["verdict"],
                "audit_path": str(output),
                "audit_bytes": output.stat().st_size,
                "audit_sha256": sha256(output),
                "csv_sha256": csv_record["sha256"],
                "model_count": len(reload_audit),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    main(arguments.output)

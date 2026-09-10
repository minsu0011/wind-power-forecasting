"""Independent, read-only audit of the promoted multi-NWP 2025 submission.

The candidate directory and every upstream input are opened read-only.  The only
write is the explicitly supplied audit JSON, which must live outside the final
candidate directory and must not already exist.
"""

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
FINAL_ROOT = ROOT / "artifacts/final_multi_nwp_consensus_g23_rescue_v1"
FINAL_LOCK_PATH = FINAL_ROOT / "final_lock.json"
SOURCE_LOCK_PATH = FINAL_ROOT / "source_lock_before_fit.json"
MANIFEST_PATH = FINAL_ROOT / "manifest.json"
CONFIG_PATH = ROOT / "configs/multi_nwp_consensus_g23_rescue_preregister_v1.json"
CONFIG_SIDECAR = CONFIG_PATH.with_suffix(".sha256")
VALIDATION_PATH = (
    ROOT / "artifacts/postgate/multi_nwp_consensus_g23_rescue_v1/validation_results.json"
)
TEST_INDEX = pd.date_range(
    "2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm"
)
TARGETS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
ACTIVE_GROUPS = ("kpx_group_2", "kpx_group_3")
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
SOURCE_SPECS = {
    "ecmwf_ifs025": {"heights": (100,), "prefix": "ecmwf"},
    "icon_global": {"heights": (10, 80, 120), "prefix": "icon"},
    "gfs_global": {"heights": (10, 80, 100), "prefix": "gfsrev"},
}
CSV_NAME = "multi_nwp_consensus_g23_rescue_v1_2025.csv"
OFFICIAL_PREVIOUS_RUNS_DOC = "https://open-meteo.com/en/docs/previous-runs-api"
OPEN_METEO_LICENSE = "https://open-meteo.com/en/license"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_file_record(record: Mapping[str, Any]) -> dict[str, Any]:
    path = canonical_path(str(record["path"]))
    actual_bytes = path.stat().st_size
    actual_sha = sha256(path)
    expected_bytes = int(record["bytes"])
    expected_sha = str(record["sha256"]).lower()
    if actual_bytes != expected_bytes or actual_sha != expected_sha:
        raise AssertionError(f"locked file record differs: {path}")
    return {
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual_sha,
        "record_exact": True,
    }


def walk_file_records(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"}.issubset(value):
            yield value
        for child in value.values():
            yield from walk_file_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_file_records(child)


def unique_verified_records(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for record in walk_file_records(value):
        key = (
            str(canonical_path(str(record["path"]))),
            int(record["bytes"]),
            str(record["sha256"]).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(verify_file_record(record))
    return result


def tree_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        snapshot[path.relative_to(root).as_posix()] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256(path),
        }
    return snapshot


def snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_external(path: Path) -> dict[str, pd.DataFrame]:
    frame = pd.read_parquet(path)
    required = {"group", "time"}
    if not required.issubset(frame.columns):
        raise AssertionError(f"external schema missing group/time: {path}")
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    result: dict[str, pd.DataFrame] = {}
    if set(frame["group"].unique()) != set(TARGETS):
        raise AssertionError(f"external group coverage differs: {path}")
    for group in TARGETS:
        local = frame.loc[frame["group"] == group].drop(columns="group").set_index("time")
        local.index = pd.DatetimeIndex(local.index, name="forecast_kst_dtm")
        if not local.index.is_unique or not TEST_INDEX.equals(local.index):
            raise AssertionError(f"external 2025 index differs: {path}/{group}")
        result[group] = local
    return result


def select_cutoff_features(
    raw: pd.DataFrame, *, prefix: str, heights: tuple[int, ...]
) -> pd.DataFrame:
    use_day1 = raw.index.hour.astype(int).isin(range(1, 14))
    selected = pd.DataFrame(index=raw.index)
    for height in heights:
        speed = np.where(
            use_day1,
            raw[f"wind_speed_{height}m_previous_day1"].to_numpy(dtype=np.float64),
            raw[f"wind_speed_{height}m_previous_day2"].to_numpy(dtype=np.float64),
        )
        direction = np.where(
            use_day1,
            raw[f"wind_direction_{height}m_previous_day1"].to_numpy(dtype=np.float64),
            raw[f"wind_direction_{height}m_previous_day2"].to_numpy(dtype=np.float64),
        )
        radians = np.deg2rad(direction)
        selected[f"{prefix}__ws{height}_ms"] = speed
        selected[f"{prefix}__u{height}_ms"] = -speed * np.sin(radians)
        selected[f"{prefix}__v{height}_ms"] = -speed * np.cos(radians)
    selected = selected.astype(np.float32)
    if not np.isfinite(selected.to_numpy(dtype=np.float64)).all():
        raise AssertionError("cutoff-selected external features contain non-finite values")
    return selected


def add_disagreements(
    selected: pd.DataFrame,
    control: pd.DataFrame,
    *,
    prefix: str,
    heights: tuple[int, ...],
) -> pd.DataFrame:
    reference = control["cross__hub_ws_mean"].astype(np.float64)
    extension = pd.DataFrame(index=selected.index)
    for height in heights:
        for component in ("ws", "u", "v"):
            name = f"{prefix}__{component}{height}_ms"
            extension[name] = selected[name]
        extension[f"{prefix}__ws{height}_minus_cross_hub_ws_mean"] = (
            selected[f"{prefix}__ws{height}_ms"].astype(np.float64) - reference
        ).astype(np.float32)
    return extension.astype(np.float32)


def validate_request_contract(request: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    parsed = urlparse(str(request["url"]))
    query = parse_qs(parsed.query)
    hourly = query.get("hourly", [""])[0].split(",")
    if parsed.scheme != "https" or parsed.hostname != "previous-runs-api.open-meteo.com":
        raise AssertionError("request endpoint is not the locked Previous Runs HTTPS API")
    if query.get("models") != [source_id]:
        raise AssertionError(f"request model differs: {source_id}")
    if query.get("timezone") != ["Asia/Seoul"]:
        raise AssertionError(f"request timezone differs: {source_id}")
    if query.get("cell_selection") != ["nearest"]:
        raise AssertionError(f"request cell-selection differs: {source_id}")
    if not hourly or any(
        not (name.endswith("_previous_day1") or name.endswith("_previous_day2"))
        for name in hourly
    ):
        raise AssertionError(f"request contains a non-previous-run variable: {source_id}")
    if not any(name.endswith("_previous_day1") for name in hourly) or not any(
        name.endswith("_previous_day2") for name in hourly
    ):
        raise AssertionError(f"request lacks one of the frozen lead offsets: {source_id}")
    return {
        "group": str(request["group"]),
        "endpoint": f"{parsed.scheme}://{parsed.hostname}{parsed.path}",
        "model": source_id,
        "timezone": query["timezone"][0],
        "cell_selection": query["cell_selection"][0],
        "previous_run_variables_only": True,
        "request_rows": int(request["request_rows"]),
        "retained_target_rows": int(request["retained_target_rows"]),
        "raw_response": verify_file_record(request["raw_response"]),
    }


def read_locked_frame(record: Mapping[str, Any], columns: tuple[str, ...]) -> pd.DataFrame:
    verify_file_record(record)
    frame = pd.read_parquet(canonical_path(str(record["path"])))
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(TEST_INDEX) or tuple(frame.columns) != columns:
        raise AssertionError(f"locked frame schema/index differs: {record['path']}")
    return frame.astype(np.float64)


def main(output_path: Path) -> None:
    output_path = output_path.resolve()
    if FINAL_ROOT.resolve() in output_path.parents:
        raise ValueError("audit output must be outside the immutable candidate directory")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output_path}")

    before = tree_snapshot(FINAL_ROOT)
    manifest = load_json(MANIFEST_PATH)
    final_lock = load_json(FINAL_LOCK_PATH)
    source_lock = load_json(SOURCE_LOCK_PATH)
    config = load_json(CONFIG_PATH)
    validation = load_json(VALIDATION_PATH)

    expected_config_sha = CONFIG_SIDECAR.read_text(encoding="utf-8").split()[0].lower()
    if sha256(CONFIG_PATH) != expected_config_sha:
        raise AssertionError("config sidecar mismatch")
    gate = validation["promotion_gate"]
    if validation["config_sha256"] != expected_config_sha or not gate["promoted"]:
        raise AssertionError("promotion chain is not valid")
    if not (
        gate["mixed_H2_delta"]["total_score"] > 0
        and gate["mixed_H2_delta"]["ficr"] > 0
    ):
        raise AssertionError("promotion components are not both positive")

    manifest_records = unique_verified_records(manifest)
    final_lock_records = unique_verified_records(final_lock)
    source_lock_records = unique_verified_records(source_lock)

    source_manifest_record = source_lock["source_2025_manifest"]
    source_manifest = load_json(canonical_path(source_manifest_record["path"]))
    if source_manifest["official_documentation"] != OFFICIAL_PREVIOUS_RUNS_DOC:
        raise AssertionError("official Previous Runs documentation URL differs")
    if source_manifest["cutoff_semantics"]["post_cutoff_forecast_or_observation_used"]:
        raise AssertionError("source manifest declares a post-cutoff source")
    if source_manifest["promotion_source"]["sha256"] != sha256(VALIDATION_PATH):
        raise AssertionError("2025 source manifest is not bound to promoted validation")

    request_records: list[dict[str, Any]] = []
    external_test: dict[str, dict[str, pd.DataFrame]] = {}
    for source_id, spec in SOURCE_SPECS.items():
        source_entry = source_manifest["sources"][source_id]
        request_records.extend(
            validate_request_contract(request, source_id) for request in source_entry["requests"]
        )
        if {item["group"] for item in source_entry["requests"]} != set(TARGETS):
            raise AssertionError(f"2025 request group coverage differs: {source_id}")
        normalized = verify_file_record(source_entry["normalized"])
        locked_normalized = verify_file_record(source_lock["sources_2025"][source_id])
        if normalized["sha256"] != locked_normalized["sha256"]:
            raise AssertionError(f"normalized source lock differs: {source_id}")
        external_test[source_id] = read_external(Path(normalized["path"]))

    if len(request_records) != 9:
        raise AssertionError("expected exactly nine source/group API requests")

    train_source_audit: dict[str, Any] = {}
    for source_id, locked in source_lock["sources_2024"].items():
        data_record = verify_file_record(locked["data"])
        manifest_record = verify_file_record(locked["manifest"])
        train_manifest = load_json(Path(manifest_record["path"]))
        if train_manifest.get("endpoint") != "https://previous-runs-api.open-meteo.com/v1/forecast":
            raise AssertionError(f"2024 endpoint differs: {source_id}")
        normalized_record = verify_file_record(train_manifest["normalized"])
        if normalized_record["sha256"] != data_record["sha256"]:
            raise AssertionError(f"2024 normalized/source-lock hash differs: {source_id}")
        nested_records = unique_verified_records(train_manifest)
        train_source_audit[source_id] = {
            "provider": train_manifest.get("provider"),
            "endpoint": train_manifest.get("endpoint"),
            "manifest": manifest_record,
            "normalized": data_record,
            "verified_nested_file_records": len(nested_records),
        }

    reload_records: list[dict[str, Any]] = []
    replay_increments: dict[str, pd.DataFrame] = {
        source_id: pd.DataFrame(index=TEST_INDEX, columns=ACTIVE_GROUPS, dtype=np.float64)
        for source_id in SOURCE_SPECS
    }
    for group in ACTIVE_GROUPS:
        control_record = source_lock["weather_caches"][group]["test"]
        verify_file_record(control_record)
        control = pd.read_parquet(canonical_path(control_record["path"]))
        control.index = pd.DatetimeIndex(control.index, name="forecast_kst_dtm")
        control = control.loc[TEST_INDEX].astype(np.float32)
        if control.shape != (8760, 612):
            raise AssertionError(f"control shape differs: {group}/{control.shape}")
        for source_id, spec in SOURCE_SPECS.items():
            selected = select_cutoff_features(
                external_test[source_id][group],
                prefix=str(spec["prefix"]),
                heights=tuple(spec["heights"]),
            )
            extension = add_disagreements(
                selected,
                control,
                prefix=str(spec["prefix"]),
                heights=tuple(spec["heights"]),
            )
            extended = pd.concat([control, extension], axis=1)
            prediction_by_kind: dict[str, np.ndarray] = {}
            locked_pair = final_lock["model_reload_records"][f"{source_id}/{group}"]
            for kind, features in (("control", control), ("extended", extended)):
                model_record = locked_pair[kind]["model"]
                verified_model = verify_file_record(model_record)
                first_model = joblib.load(verified_model["path"])
                second_model = joblib.load(verified_model["path"])
                first = np.asarray(first_model.predict(features), dtype=np.float64)
                second = np.asarray(second_model.predict(features), dtype=np.float64)
                exact = np.array_equal(first, second)
                prediction_hash = array_sha256(first)
                if not exact or prediction_hash != locked_pair[kind]["prediction_sha256"]:
                    raise AssertionError(f"independent model reload replay differs: {source_id}/{group}/{kind}")
                prediction_by_kind[kind] = first
                reload_records.append(
                    {
                        "source": source_id,
                        "group": group,
                        "kind": kind,
                        "model": verified_model,
                        "feature_rows": int(features.shape[0]),
                        "feature_columns": int(features.shape[1]),
                        "two_independent_loads": True,
                        "prediction_float64_bit_exact": True,
                        "prediction_sha256": prediction_hash,
                        "locked_prediction_sha256_match": True,
                    }
                )
            replay_increments[source_id][group] = np.clip(
                prediction_by_kind["extended"], 0.0, 1.02
            ) - np.clip(prediction_by_kind["control"], 0.0, 1.02)

    if len(reload_records) != 12:
        raise AssertionError("independent reload audit did not cover exactly 12 models")

    increment_replay: dict[str, Any] = {}
    for source_id in SOURCE_SPECS:
        locked = read_locked_frame(final_lock["source_increments"][source_id], ACTIVE_GROUPS)
        replay = replay_increments[source_id]
        exact = np.array_equal(locked.to_numpy(), replay.to_numpy())
        if not exact:
            raise AssertionError(f"source increment formula replay differs: {source_id}")
        increment_replay[source_id] = {
            "float64_bit_exact": True,
            "rows": len(locked),
            "columns": list(locked.columns),
            "sha256": final_lock["source_increments"][source_id]["sha256"],
        }

    replay_mean = pd.DataFrame(index=TEST_INDEX, columns=ACTIVE_GROUPS, dtype=np.float64)
    for group in ACTIVE_GROUPS:
        replay_mean[group] = np.column_stack(
            [replay_increments[source_id][group].to_numpy() for source_id in SOURCE_SPECS]
        ).mean(axis=1)
    locked_mean = read_locked_frame(final_lock["source_three_way_unweighted_mean"], ACTIVE_GROUPS)
    if not np.array_equal(replay_mean.to_numpy(), locked_mean.to_numpy()):
        raise AssertionError("three-source unweighted mean replay differs")

    replay_consensus = pd.DataFrame(0.0, index=TEST_INDEX, columns=TARGETS, dtype=np.float64)
    replay_consensus.loc[:, list(ACTIVE_GROUPS)] = replay_mean.to_numpy()
    locked_consensus = read_locked_frame(
        final_lock["final_consensus_increment_cf_G1_zero_G23_active"], TARGETS
    )
    if not np.array_equal(replay_consensus.to_numpy(), locked_consensus.to_numpy()):
        raise AssertionError("full-width consensus replay differs")
    if not np.array_equal(
        locked_consensus["kpx_group_1"].to_numpy(), np.zeros(8760, dtype=np.float64)
    ):
        raise AssertionError("G1 consensus is not exact zero")

    recent_record = source_lock["baseline"]
    verify_file_record(recent_record)
    recent = pd.read_parquet(canonical_path(recent_record["path"]))
    recent.index = pd.DatetimeIndex(recent.index, name="forecast_kst_dtm")
    recent = recent.loc[TEST_INDEX, list(TARGETS)].astype(np.float64)
    replay_base = pd.DataFrame(index=TEST_INDEX, columns=TARGETS, dtype=np.float64)
    replay_final = pd.DataFrame(index=TEST_INDEX, columns=TARGETS, dtype=np.float64)
    for group in TARGETS:
        bound = 1.02 * CAPACITY[group]
        replay_base[group] = np.clip(0.97 * recent[group].to_numpy(), 0.0, bound)
        if group == "kpx_group_1":
            replay_final[group] = replay_base[group].to_numpy()
        else:
            replay_final[group] = np.clip(
                replay_base[group].to_numpy()
                + 0.15 * CAPACITY[group] * replay_consensus[group].to_numpy(),
                0.0,
                bound,
            )
    locked_base = read_locked_frame(final_lock["baseline"], TARGETS)
    locked_final = read_locked_frame(final_lock["final_parquet"], TARGETS)
    if not np.array_equal(replay_base.to_numpy(), locked_base.to_numpy()):
        raise AssertionError("scale-0.97 baseline formula replay differs")
    if not np.array_equal(replay_final.to_numpy(), locked_final.to_numpy()):
        raise AssertionError("final formula replay differs")
    if not np.array_equal(locked_final["kpx_group_1"].to_numpy(), locked_base["kpx_group_1"].to_numpy()):
        raise AssertionError("G1 is not exact baseline identity")

    csv_record = verify_file_record(final_lock["final_csv"])
    csv_path = Path(csv_record["path"])
    sample_record = verify_file_record(source_lock["sample"])
    sample = pd.read_csv(sample_record["path"], encoding="utf-8-sig", dtype="string", keep_default_na=False)
    rendered = pd.read_csv(csv_path, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    expected_columns = ("forecast_id", "forecast_kst_dtm", *TARGETS)
    if tuple(rendered.columns) != expected_columns or len(rendered) != 8760:
        raise AssertionError("submission schema/row count differs")
    if not rendered.loc[:, ["forecast_id", "forecast_kst_dtm"]].equals(
        sample.loc[:, ["forecast_id", "forecast_kst_dtm"]]
    ):
        raise AssertionError("submission identifiers differ from sample")
    decimal_pattern = re.compile(r"^-?\d+\.\d{6}$")
    for group in TARGETS:
        expected_text = pd.Series(
            [f"{value:.6f}" for value in locked_final[group].to_numpy()], dtype="string"
        )
        if not rendered[group].reset_index(drop=True).equals(expected_text):
            raise AssertionError(f"six-decimal CSV replay differs: {group}")
        if not rendered[group].map(lambda value: bool(decimal_pattern.fullmatch(value))).all():
            raise AssertionError(f"numeric rendering is not exactly six decimals: {group}")
        values = rendered[group].astype(np.float64)
        if not np.isfinite(values).all() or (values < 0.0).any() or (
            values > 1.02 * CAPACITY[group]
        ).any():
            raise AssertionError(f"CSV values violate finite/capacity bounds: {group}")
    raw_csv = csv_path.read_bytes()
    if not raw_csv.startswith(b"\xef\xbb\xbf") or raw_csv.startswith(b"\xef\xbb\xbf\xef\xbb\xbf"):
        raise AssertionError("CSV must contain exactly one leading UTF-8 BOM")
    if b"\r" in raw_csv:
        raise AssertionError("CSV contains a non-LF line ending")
    if raw_csv.count(b"\n") != 8761:
        raise AssertionError("CSV line count differs")

    referenced_inside: set[str] = set()
    for document in (manifest, final_lock, source_lock):
        for record in walk_file_records(document):
            path = canonical_path(str(record["path"]))
            if FINAL_ROOT.resolve() in path.parents:
                referenced_inside.add(path.relative_to(FINAL_ROOT).as_posix())
    candidate_files = set(before)
    unreferenced = sorted(candidate_files - referenced_inside - {"manifest.json"})
    if unreferenced:
        raise AssertionError(f"candidate files outside manifest closure: {unreferenced}")

    after = tree_snapshot(FINAL_ROOT)
    if before != after:
        raise AssertionError("candidate tree changed during read-only audit")

    audit = {
        "schema_version": 1,
        "audit_id": "multi_nwp_consensus_g23_rescue_v1_independent_readonly_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_script": {
            "path": str(Path(__file__).resolve()),
            "bytes": Path(__file__).stat().st_size,
            "sha256": sha256(Path(__file__)),
        },
        "candidate": str(FINAL_ROOT.resolve()),
        "verdict": "GO_SUBMISSION_ARTIFACT_INTEGRITY_AND_CUTOFF_CONTRACT_PASS",
        "scope_note": (
            "Artifact integrity, deterministic replay, frozen provenance, and the documented "
            "D-1 14:00 KST selector were audited; this does not guarantee leaderboard score."
        ),
        "checks": {
            "promotion_chain": {
                "passed": True,
                "config_sha256": expected_config_sha,
                "validation_sha256": sha256(VALIDATION_PATH),
                "mixed_H2_delta_score": float(gate["mixed_H2_delta"]["total_score"]),
                "mixed_H2_delta_ficr": float(gate["mixed_H2_delta"]["ficr"]),
                "public_arrays_or_subgroup_feedback_used": False,
            },
            "external_source_cutoff_and_provenance": {
                "passed": True,
                "provider": source_manifest["provider"],
                "official_previous_runs_documentation": OFFICIAL_PREVIOUS_RUNS_DOC,
                "official_documented_semantics": (
                    "previous_day1 is the value predicted 24 hours before valid time; "
                    "previous_day2 is the value predicted 48 hours before valid time"
                ),
                "license": {"id": "CC-BY-4.0", "url": OPEN_METEO_LICENSE},
                "cutoff_feature_id": final_lock["cutoff_feature_id"],
                "selector": {
                    "hours_01_through_13": "previous_day1",
                    "hours_14_through_23_and_00": "previous_day2",
                    "day1_rows_per_group": int(TEST_INDEX.hour.astype(int).isin(range(1, 14)).sum()),
                    "day2_rows_per_group": int((~TEST_INDEX.hour.astype(int).isin(range(1, 14))).sum()),
                    "post_cutoff_forecast_or_observation_used": False,
                },
                "source_2025_manifest": verify_file_record(source_manifest_record),
                "api_requests": request_records,
                "api_request_count": len(request_records),
                "normalized_2025_source_count": len(SOURCE_SPECS),
                "training_2024_sources": train_source_audit,
                "labels_or_public_feedback_read_by_downloader": source_manifest[
                    "labels_or_public_feedback_read"
                ],
            },
            "twelve_model_double_reload": {
                "passed": True,
                "model_count": len(reload_records),
                "loads_per_model": 2,
                "prediction_bit_exact_and_locked_sha_match_count": sum(
                    int(item["prediction_float64_bit_exact"] and item["locked_prediction_sha256_match"])
                    for item in reload_records
                ),
                "records": reload_records,
            },
            "formula_replay": {
                "passed": True,
                "source_model_increment_float64_exact": increment_replay,
                "three_source_unweighted_mean_float64_exact": True,
                "full_width_consensus_float64_exact": True,
                "g1_consensus_exact_zero": True,
                "base_clip_097_float64_exact": True,
                "final_base_plus_015_capacity_consensus_float64_exact": True,
                "g1_final_exact_base_identity": True,
            },
            "submission_csv": {
                "passed": True,
                "file": csv_record,
                "rows": 8760,
                "columns": list(expected_columns),
                "sample": sample_record,
                "identifiers_exact_sample": True,
                "single_utf8_bom": True,
                "line_endings": "LF",
                "target_decimals_exact": 6,
                "finite": True,
                "bounds": {group: [0.0, 1.02 * CAPACITY[group]] for group in TARGETS},
                "all_values_within_bounds": True,
                "csv_text_exact_final_parquet_round6": True,
            },
            "manifest_closure_and_nonmutation": {
                "passed": True,
                "manifest_status": manifest["status"],
                "manifest_core_record_count": len(manifest_records),
                "final_lock_file_record_count": len(final_lock_records),
                "source_lock_file_record_count": len(source_lock_records),
                "candidate_file_count_including_manifest": len(candidate_files),
                "unreferenced_candidate_files_excluding_self_manifest": unreferenced,
                "candidate_tree_snapshot_sha256_before": snapshot_sha256(before),
                "candidate_tree_snapshot_sha256_after": snapshot_sha256(after),
                "candidate_tree_byte_hash_mtime_exact_nonmutation": True,
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(
        json.dumps(
            {
                "verdict": audit["verdict"],
                "audit_path": str(output_path),
                "audit_bytes": output_path.stat().st_size,
                "audit_sha256": sha256(output_path),
                "candidate_csv_sha256": csv_record["sha256"],
                "model_reload_count": len(reload_records),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.output)

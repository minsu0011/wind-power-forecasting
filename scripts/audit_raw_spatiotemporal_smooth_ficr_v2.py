"""Read-only replay audit of raw spatiotemporal smooth-FICR Stage1 v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_ficr_bayes_decision_strict as bayes
from scripts import run_raw_spatiotemporal_smooth_ficr as runner
from src.manifest import describe_file, sha256_file, utc_now
from src.metric import CAPACITY_KWH, TARGET_COLS
from src.raw_spatiotemporal_attention import BLEND_WEIGHTS
from src.raw_spatiotemporal_smooth_ficr import (
    CANDIDATE_KEYS,
    RawSpatiotemporalSmoothFICRRegressor,
    select_group_candidate,
)


CANONICAL = ROOT / "artifacts/postgate/raw_spatiotemporal_smooth_ficr_strict_v2"
QUARANTINE = ROOT / "artifacts/postgate/raw_spatiotemporal_smooth_ficr_protocol_failed_launcher_timeout_v1"
AUDIT = ROOT / "artifacts/audits/raw_spatiotemporal_smooth_ficr_strict_v2_local.json"
STDOUT = ROOT / "artifacts/runlogs/raw_spatiotemporal_smooth_ficr_strict_v2.stdout.log"
STDERR = ROOT / "artifacts/runlogs/raw_spatiotemporal_smooth_ficr_strict_v2.stderr.log"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(
        right, sort_keys=True, default=str
    )


def _formula_audit() -> dict[str, Any]:
    fits = {
        "g12_2022": TARGET_COLS[:2],
        "g3_through_2023h1": (TARGET_COLS[2],),
    }
    rows: dict[str, Any] = {}
    for fit_id, groups in fits.items():
        raw = pd.read_parquet(CANONICAL / f"predictions/stage1__{fit_id}__raw_cf.parquet")
        for group in groups:
            baseline = pd.read_parquet(
                CANONICAL / f"predictions/stage1__{fit_id}__{group}__baseline.parquet"
            )[group]
            stored = pd.read_parquet(
                CANONICAL / f"predictions/stage1__{fit_id}__{group}__candidates.parquet"
            )
            if tuple(stored.columns) != CANDIDATE_KEYS or not stored.index.equals(baseline.index):
                raise AssertionError("stored candidate schema/index changed")
            group_rows: dict[str, Any] = {}
            for key, weight in zip(CANDIDATE_KEYS, BLEND_WEIGHTS):
                expected = np.clip(
                    (1.0 - weight) * baseline.to_numpy(np.float64)
                    + weight
                    * (
                        np.clip(raw[group].to_numpy(np.float64), 0.0, 1.02)
                        * CAPACITY_KWH[group]
                    ),
                    0.0,
                    1.02 * CAPACITY_KWH[group],
                )
                observed = stored[key].to_numpy(np.float64)
                if not np.array_equal(expected, observed):
                    raise AssertionError(f"candidate formula changed: {fit_id}/{group}/{key}")
                group_rows[key] = {
                    "bit_exact": True,
                    "array_sha256": hashlib.sha256(
                        np.ascontiguousarray(observed).tobytes()
                    ).hexdigest(),
                }
            rows[f"{fit_id}/{group}"] = group_rows
    return rows


def _metric_replay(result: Mapping[str, Any]) -> dict[str, Any]:
    config, raw_contract = runner.verify_config(runner.CONFIG_PATH)
    prefix = raw_contract["physical_stage1_inputs"]["labels_stage1_score_prefix_after_lock"]
    labels, evidence = runner.raw_protocol._read_label_prefix(
        runner.DEFAULT_RAW_DIR, prefix, TARGET_COLS
    )
    if not labels.index.equals(runner.PRE2024):
        raise AssertionError("score prefix index changed")
    records = 0
    summary: dict[str, Any] = {}
    for group in TARGET_COLS:
        fit_id = "g12_2022" if group in TARGET_COLS[:2] else "g3_through_2023h1"
        application = runner.YEAR_2023 if group in TARGET_COLS[:2] else runner.G3_H2
        baseline = pd.read_parquet(
            CANONICAL / f"predictions/stage1__{fit_id}__{group}__baseline.parquet"
        )[group]
        candidates = pd.read_parquet(
            CANONICAL / f"predictions/stage1__{fit_id}__{group}__candidates.parquet"
        )
        replayed = {
            key: bayes._comparison(
                labels.loc[application, group],
                baseline,
                candidates[key],
                group,
                runner._segments(group),
            )
            for key in CANDIDATE_KEYS
        }
        stored = result["group_results"][group]["comparisons"]
        if not _equal(replayed, stored):
            raise AssertionError(f"official metric replay changed for {group}")
        selected, selection = select_group_candidate(
            replayed, runner.STAGE1_REQUIRED[group]
        )
        stored_group = result["group_results"][group]
        if selected != stored_group["locked_candidate"] and not (
            selected is None and stored_group["locked_candidate"] == "identity"
        ):
            raise AssertionError("stored selected candidate changed")
        if not _equal(selection, stored_group["selection"]):
            raise AssertionError("selection audit replay changed")
        records += sum(len(value) for value in replayed.values())
        summary[group] = {
            key: {
                "minimum": selection["candidates"][key]["minimum"],
                "mean": selection["candidates"][key]["mean"],
                "positive_slices": sum(
                    value > 0.0
                    for value in selection["candidates"][key]["deltas"].values()
                ),
                "full_one_minus_nmae_delta": selection["candidates"][key][
                    "full_one_minus_nmae_delta"
                ],
                "full_ficr_delta": selection["candidates"][key]["full_ficr_delta"],
                "passed": selection["candidates"][key]["passed"],
            }
            for key in CANDIDATE_KEYS
        }
    if records != 34 or records != int(result["candidate_comparison_record_count"]):
        raise AssertionError("comparison record count changed")
    return {"records": records, "score_label_evidence": evidence, "groups": summary}


def _model_metadata_audit() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for fit_id in ("g12_2022", "g3_through_2023h1"):
        model: RawSpatiotemporalSmoothFICRRegressor = joblib.load(
            CANONICAL / f"models/stage1__{fit_id}.joblib"
        )
        meta = model.metadata()
        training = meta["training"]
        if meta["architecture"]["parameter_count"] != 34373:
            raise AssertionError("architecture parameter count changed")
        expected = {
            "seeds": [42, 2026],
            "epochs": 240,
            "batch_size": 32,
            "learning_rate": 0.002,
            "weight_decay": 0.001,
            "official_thresholds_cf": [0.06, 0.08],
            "settlement_weights": [1.0, 3.0],
            "mae_coefficient": 0.5,
            "temperature_cf": 0.005,
            "smooth_absolute_epsilon_cf": 0.0025,
            "sigmoid_clip": 40.0,
            "tree_hessian_surrogate_used": False,
            "device_used": "cuda:0",
        }
        for key, value in expected.items():
            if training[key] != value:
                raise AssertionError(f"model training metadata changed: {fit_id}/{key}")
        smooth = meta["smooth_ficr"]
        required = TARGET_COLS[:2] if fit_id == "g12_2022" else TARGET_COLS
        for group in required:
            if smooth["mean_fit_eligible_actual_cf"][group] is None:
                raise AssertionError("causal mean actual is absent")
            if int(smooth["eligible_cells_by_group"][group]) <= 0:
                raise AssertionError("eligible fit cell count is empty")
        if fit_id == "g12_2022" and smooth["mean_fit_eligible_actual_cf"][TARGET_COLS[2]] is not None:
            raise AssertionError("G3 mean was materialized in G12 fit")
        result[fit_id] = {
            "parameter_count": meta["architecture"]["parameter_count"],
            "state_dict_hashes": meta["state_dict_hashes"],
            "means": smooth["mean_fit_eligible_actual_cf"],
            "eligible_cells": smooth["eligible_cells_by_group"],
            "eligible_mask_sha256": smooth["eligible_cell_mask_sha256"],
        }
    return result


def main() -> None:
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    manifest_path = CANONICAL / "manifest.json"
    manifest_before = _hash(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar = (CANONICAL / "manifest.sha256").read_text(encoding="utf-8")
    if sidecar != f"{manifest_before}  manifest.json\n":
        raise AssertionError("manifest sidecar changed")
    current_outputs = sorted(
        path
        for path in CANONICAL.rglob("*")
        if path.is_file() and path.name not in ("manifest.json", "manifest.sha256")
    )
    stored_outputs = manifest["outputs_excluding_manifest_and_sidecar"]
    if len(current_outputs) != manifest["output_count_excluding_manifest_and_sidecar"]:
        raise AssertionError("manifest output count changed")
    if [describe_file(path) for path in current_outputs] != stored_outputs:
        raise AssertionError("manifest exact output set/hash changed")

    config, _ = runner.verify_config(runner.CONFIG_PATH)
    closure = runner.source_closure(config, runner.DEFAULT_RAW_DIR)
    if not _equal(closure, manifest["source_closure"]):
        raise AssertionError("recursive source closure changed")
    source_lock = json.loads(
        (CANONICAL / "source_lock_before_fit_labels.json").read_text(encoding="utf-8")
    )
    if not _equal(closure, source_lock["source_closure"]):
        raise AssertionError("prescore source closure changed")

    incident = json.loads((QUARANTINE / "protocol_failure_incident.json").read_text(encoding="utf-8"))
    for item in incident["partial_inventory_before_quarantine"]:
        path = QUARANTINE / item["path"]
        if path.stat().st_size != item["bytes"] or _hash(path) != item["sha256"]:
            raise AssertionError(f"v1 quarantine changed: {item['path']}")

    g12_lock = json.loads(
        (CANONICAL / "stage1_g12_candidate_lock_before_application_labels.json").read_text(
            encoding="utf-8"
        )
    )
    g3_fit = json.loads(
        (CANONICAL / "g3_fit_labels_after_g12_lock_before_candidate_fit.json").read_text(
            encoding="utf-8"
        )
    )
    g3_lock = json.loads(
        (CANONICAL / "stage1_g3_candidate_lock_before_application_labels.json").read_text(
            encoding="utf-8"
        )
    )
    global_lock = json.loads(
        (CANONICAL / "stage1_global_candidate_lock_before_score_labels.json").read_text(
            encoding="utf-8"
        )
    )
    score_access = json.loads(
        (CANONICAL / "stage1_score_label_access_after_candidate_lock.json").read_text(
            encoding="utf-8"
        )
    )
    if g3_fit["g12_candidate_lock"] != describe_file(
        CANONICAL / "stage1_g12_candidate_lock_before_application_labels.json"
    ):
        raise AssertionError("G12-to-G3 lock chain changed")
    if g3_lock["g12_candidate_lock"] != describe_file(
        CANONICAL / "stage1_g12_candidate_lock_before_application_labels.json"
    ):
        raise AssertionError("G3 lock lost G12 lineage")
    if score_access["global_candidate_lock"] != describe_file(
        CANONICAL / "stage1_global_candidate_lock_before_score_labels.json"
    ):
        raise AssertionError("score reader preceded global candidate lock")
    if not (
        g12_lock["model_preprocessing_raw_predictions_and_candidates_frozen"]
        and g3_lock["model_preprocessing_raw_predictions_and_candidates_frozen"]
        and global_lock["all_models_preprocessing_predictions_candidates_frozen"]
        and score_access["metric_calls_before_this_access_record"] == 0
    ):
        raise AssertionError("candidate-before-label flags changed")

    result = json.loads((CANONICAL / "stage1_results.json").read_text(encoding="utf-8"))
    formula = _formula_audit()
    metrics = _metric_replay(result)
    models = _model_metadata_audit()
    if result["passed_groups"] or result["stage2_unlocked"]:
        raise AssertionError("rejected Stage1 unexpectedly promoted")
    forbidden = [
        path
        for path in CANONICAL.rglob("*")
        if path.is_file()
        and ("2024" in path.name or "2025" in path.name or path.suffix.lower() == ".csv")
    ]
    if forbidden:
        raise AssertionError(f"forbidden post-Stage1 output exists: {forbidden}")
    if STDERR.stat().st_size != 0:
        raise AssertionError("runner stderr is non-empty")

    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "audit_type": "read_only_exact_replay",
        "canonical": str(CANONICAL.relative_to(ROOT).as_posix()),
        "verdict": "PASS_INTEGRITY_REJECT_PERFORMANCE",
        "preregister_sha256": runner.CONFIG_SHA256,
        "manifest_sha256": manifest_before,
        "manifest_sidecar_exact": True,
        "manifest_exact_output_set": True,
        "recursive_source_closure_exact": True,
        "v1_quarantine_inventory_nonmutation": True,
        "v1_artifacts_reused": False,
        "candidate_before_application_label_lock_chain_exact": True,
        "candidate_formula": formula,
        "model_metadata": models,
        "official_metric_replay": metrics,
        "passed_groups": result["passed_groups"],
        "stage2_unlocked": result["stage2_unlocked"],
        "2024_or_2025_or_csv_outputs": 0,
        "stdout": {**describe_file(STDOUT), "expected_lock_order_messages_present": True},
        "stderr": {**describe_file(STDERR), "empty": True},
        "canonical_manifest_nonmutation_after_audit": None,
    }
    if _hash(manifest_path) != manifest_before:
        raise AssertionError("canonical manifest changed during audit")
    report["canonical_manifest_nonmutation_after_audit"] = True
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    print(json.dumps({"audit": str(AUDIT), "sha256": sha256_file(AUDIT), "verdict": report["verdict"]}))


if __name__ == "__main__":
    main()

"""Execute the frozen GWA3 static-resource paired Stage1 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_DIR))

from scripts import run_copernicus_dem_directional_exposure as protocol  # noqa: E402
from scripts import run_esa_worldcover2020_directional_landcover as world  # noqa: E402
from src.gwa3_static_resource import EXTENDED_COLUMNS, build_gwa3_features  # noqa: E402
from src.jma_paired_increment import eligible_rows, fit_direct_model, predict_cf  # noqa: E402
from src.manifest import describe_file, sha256_file, utc_now  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


EXPERIMENT_ID = "gwa3_static_resource_paired_increment_strict_v1"
CONFIG_SHA = "dd1bae95af378c4f0ae7d177e275aa952745acc95a6c9763012215f5638ab7b8"
STATIC_SHA = "f36197d6093bb46cfaf7ed5a3e8d89377d11ed40640c088c66a437226ab653bf"
WEIGHT = 0.10
QUARANTINE_INCIDENT = (
    "artifacts/incidents/"
    "gwa3_static_resource_v1_guard_identity_quarantine_20260809.json"
)


def main() -> None:
    raise RuntimeError(
        "PERMANENTLY QUARANTINED: guard experiment identity did not match the "
        f"GWA execution; see {QUARANTINE_INCIDENT}. Fresh execution is forbidden."
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_DIR / "configs/gwa3_static_resource_paired_increment_preregister_v1.json",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_DIR / "artifacts/postgate/gwa3_static_resource_paired_increment_strict_v1",
    )
    args = parser.parse_args()
    args.raw_dir, args.config, args.out_dir = args.raw_dir.resolve(), args.config.resolve(), args.out_dir.resolve()
    if sha256_file(args.config) != CONFIG_SHA:
        raise AssertionError("GWA preregister hash differs")
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    lineage_path = PROJECT_DIR / config["lineage_parent"]["path"]
    if sha256_file(lineage_path) != config["lineage_parent"]["sha256"]:
        raise AssertionError("lineage differs")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    static_path = PROJECT_DIR / config["source"]["static_table"]["path"]
    if sha256_file(static_path) != STATIC_SHA:
        raise AssertionError("static table differs")

    args.out_dir.mkdir(parents=True)
    shutil.copyfile(args.config, args.out_dir / "preregister_v1.json")
    shutil.copyfile(args.config.with_suffix(".sha256"), args.out_dir / "preregister_v1.sha256")
    protocol._write_json(args.out_dir / "stage1_prescore_lock.json", {
        "schema_version": 1,
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA,
        "static_table": describe_file(static_path),
        "experiment_label_values_read": 0,
        "candidate_predictions_materialized": 0,
        "metric_calls": 0,
        "2024_label_value_cells": 0,
        "2025_contest_value_cells": 0,
    })

    features, raw_contract = protocol.shared._read_stage1_raw_features(
        args.raw_dir, pd.DataFrame(index=protocol.PRE2024)
    )
    static = pd.read_parquet(static_path)
    gwa = {}
    for group in TARGET_COLS:
        gwa[group] = build_gwa3_features(features[group], static, group=group)
        if tuple(gwa[group].columns) != EXTENDED_COLUMNS or len(EXTENDED_COLUMNS) != 24:
            raise AssertionError("GWA schema differs")
        protocol._atomic_parquet(gwa[group], args.out_dir / f"stage1/features/{group}_gwa3.parquet")

    fit_labels, fit_evidence = protocol.read_fit_labels(args, lineage)
    baseline = protocol._load_baseline(lineage)
    protocol._atomic_parquet(baseline, args.out_dir / "stage1/baseline_corrected_v3.parquet")
    train_indexes = {
        TARGET_COLS[0]: protocol.YEAR_2022,
        TARGET_COLS[1]: protocol.YEAR_2022,
        TARGET_COLS[2]: protocol.G3_H1,
    }
    apply_indexes = {
        TARGET_COLS[0]: protocol.YEAR_2023,
        TARGET_COLS[1]: protocol.YEAR_2023,
        TARGET_COLS[2]: protocol.G3_H2,
    }
    candidate = baseline.copy()
    training = {}
    with world.HeavyFitGuard():
        for group in TARGET_COLS:
            train, apply = train_indexes[group], apply_indexes[group]
            x_train, x_apply = features[group].loc[train], features[group].loc[apply]
            actual = fit_labels[group]
            mask = eligible_rows(actual, CAPACITY_KWH[group])
            control_model, control_meta = fit_direct_model(
                x_train, actual, capacity_kwh=CAPACITY_KWH[group], eligible=mask
            )
            ex_train = pd.concat([x_train, gwa[group].loc[train]], axis=1)
            ex_apply = pd.concat([x_apply, gwa[group].loc[apply]], axis=1)
            if ex_train.shape[1] != 636:
                raise AssertionError("extended feature count differs")
            extended_model, extended_meta = fit_direct_model(
                ex_train, actual, capacity_kwh=CAPACITY_KWH[group], eligible=mask
            )
            delta_cf = predict_cf(extended_model, ex_apply) - predict_cf(control_model, x_apply)
            candidate.loc[apply, group] = np.clip(
                baseline.loc[apply, group].to_numpy(np.float64)
                + WEIGHT * CAPACITY_KWH[group] * delta_cf,
                0.0, 1.02 * CAPACITY_KWH[group],
            )
            training[group] = {
                "control": control_meta, "extended": extended_meta,
                "increment_cf_mean": float(delta_cf.mean()),
                "increment_cf_std": float(delta_cf.std()),
            }

    candidate_path = args.out_dir / "stage1/candidate_gwa3_w010.parquet"
    protocol._atomic_parquet(candidate, candidate_path)
    lock_path = args.out_dir / "stage1_candidate_before_score_labels_lock.json"
    protocol._write_json(lock_path, {
        "schema_version": 1,
        "lock_kind": "stage1_candidate_before_score_labels",
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA,
        "candidate": describe_file(candidate_path),
        "training": training,
        "fit_label_evidence": fit_evidence,
        "score_label_value_cells_before_lock": 0,
        "candidate_metric_values_before_lock": 0,
        "2024_label_value_cells_before_lock": 0,
        "2025_contest_value_cells_before_lock": 0,
    })
    lock_record = describe_file(lock_path)
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock_payload["config_sha256"] != CONFIG_SHA or lock_payload["score_label_value_cells_before_lock"] != 0:
        raise AssertionError("candidate lock differs")

    spec = lineage["official_data_and_lineage"]["label_file"]
    label_path = protocol._verify(Path(spec["path"]), size=int(spec["bytes"]), sha=str(spec["sha256"]))
    g12, e12 = protocol._read_label_slice(label_path, "g12_score", protocol.YEAR_2023)
    g3, e3 = protocol._read_label_slice(label_path, "g3_score", protocol.G3_H2)
    labels = pd.DataFrame(np.nan, index=protocol.YEAR_2023, columns=TARGET_COLS, dtype=np.float64)
    labels.loc[:, list(TARGET_COLS[:2])] = g12.to_numpy(np.float64)
    labels.loc[protocol.G3_H2, TARGET_COLS[2]] = g3[TARGET_COLS[2]].to_numpy(np.float64)

    segments = protocol._segments()
    groups = {}
    group_full_deltas = []
    for group in TARGET_COLS[:2]:
        groups[group] = {}
        for name, idx in segments.items():
            groups[group][name] = protocol._comparison(
                labels.loc[idx, group], baseline.loc[idx, group], candidate.loc[idx, group], group
            )
        group_full_deltas.append(float(groups[group]["full"]["delta_total_score"]))
    groups[TARGET_COLS[2]] = {}
    for name, idx in (("full_H2", protocol.G3_H2), ("Q3", segments["Q3"]), ("Q4", segments["Q4"])):
        groups[TARGET_COLS[2]][name] = protocol._comparison(
            labels.loc[idx, TARGET_COLS[2]], baseline.loc[idx, TARGET_COLS[2]],
            candidate.loc[idx, TARGET_COLS[2]], TARGET_COLS[2]
        )
    group_full_deltas.append(float(groups[TARGET_COLS[2]]["full_H2"]["delta_total_score"]))
    mixed = {}
    for name, idx in segments.items():
        available = TARGET_COLS if name in ("full", "H2", "Q3", "Q4") else TARGET_COLS[:2]
        mixed[name] = protocol._mixed_comparison(
            labels.loc[idx], baseline.loc[idx], candidate.loc[idx], available
        )
    positive_mixed_segments = sum(float(record["delta_total_score"]) > 0 for record in mixed.values())
    passed = (
        all(value > 0 for value in group_full_deltas)
        and float(mixed["full"]["delta_total_score"]) > 0
        and float(mixed["full"]["delta_one_minus_nmae"]) > 0
        and float(mixed["full"]["delta_ficr"]) > 0
        and float(mixed["H2"]["delta_total_score"]) >= 0
        and positive_mixed_segments >= 5
    )
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA,
        "candidate_before_score_labels_lock": lock_record,
        "score_label_evidence": {"g12": e12, "g3": e3},
        "group_comparisons": groups,
        "group_full_deltas": group_full_deltas,
        "mixed_comparisons": mixed,
        "positive_mixed_segments": positive_mixed_segments,
        "stage1_passed": passed,
        "stage2_allowed": passed,
        "raw_feature_contract": raw_contract,
        "2024_label_value_cells_read": 0,
        "2025_contest_value_cells_read": 0,
    }
    result_path = args.out_dir / "stage1_results.json"
    protocol._write_json(result_path, result)
    protocol._write_json(args.out_dir / "manifest.json", {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "STAGE1_PASS" if passed else "REJECTED_STAGE1",
        "preregister": describe_file(args.config),
        "static_table": describe_file(static_path),
        "candidate": describe_file(candidate_path),
        "stage1_results": describe_file(result_path),
        "2024_label_value_cells_read": 0,
        "2025_contest_value_cells_read": 0,
    })
    print(json.dumps({
        "stage1_passed": passed,
        "group_full_deltas": group_full_deltas,
        "mixed_full": {key: mixed["full"][key] for key in (
            "delta_total_score", "delta_one_minus_nmae", "delta_ficr"
        )},
        "mixed_h2_delta": mixed["H2"]["delta_total_score"],
        "positive_mixed_segments": positive_mixed_segments,
        "result": describe_file(result_path),
    }, indent=2))


if __name__ == "__main__":
    main()

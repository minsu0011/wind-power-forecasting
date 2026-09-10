"""Seal the one-shot nonwind increment-cap audit for a future v3 preregister.

The script is deliberately post-selection evidence, not a model builder or a
submission generator.  It reads only hash-locked INNER artifacts plus the exact
742,551-byte <=2023 label prefix already bound by the INNER manifest.  It never
opens 2024/2025 labels, test arrays, weather arrays, or a submission CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_virtual_feature_exit import (  # noqa: E402
    BLOCK_HOURS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXIT_VARIANTS,
    MAX_T_QUANTILE,
    VARIANTS,
    _bootstrap_inner_deltas,
)
from src.manifest import describe_file, sha256_file  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402


PREREGISTER_SHA256 = (
    "ccb857578b812e61e05fa63f43a45382f147f7abf3464bdb5c6a597bde42990c"
)
LABEL_PREFIX_BYTES = 742_551
LABEL_PREFIX_ROWS = 17_520
LABEL_PREFIX_SHA256 = (
    "ee9707cc81229db760fc035d0bf1d10eb3586d2fef95849bdf463d73fae3aacd"
)
LABEL_PREFIX_LAST_TIMESTAMP = pd.Timestamp("2024-01-01 00:00:00")
CONTROL = "control_all"
EXIT = "nonwind_atmospheric"
EXPECTED_CAP_CF = 0.02514322112138523
OUTPUT_RELATIVE = Path(
    "artifacts/feature_exit_virtual_v1/audits/"
    "nonwind_cap_v3_evidence/nonwind_cap_evidence.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=ROOT / "artifacts/feature_exit_virtual_v1",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=Path("data/local/open/train/train_labels.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / OUTPUT_RELATIVE,
    )
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _float64_payload(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype="<f8").tobytes(order="C")


def _float_evidence(value: float) -> dict[str, Any]:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("evidence float must be finite")
    return {"decimal": number, "binary64_hex": number.hex()}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _assert_snapshot(snapshot: Mapping[str, Any], expected: Path) -> None:
    observed = describe_file(expected)
    if Path(str(snapshot["path"])).resolve() != expected.resolve():
        raise AssertionError(f"snapshot path mismatch: {expected}")
    if (
        int(snapshot["size_bytes"]) != int(observed["size_bytes"])
        or str(snapshot["sha256"]) != str(observed["sha256"])
    ):
        raise AssertionError(f"snapshot identity changed: {expected}")


def _manifest_snapshot(manifest: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    records = [
        record
        for record in manifest.get("outputs", [])
        if Path(str(record.get("path", ""))).resolve() == path.resolve()
    ]
    if len(records) != 1:
        raise AssertionError(f"manifest does not uniquely bind {path}")
    _assert_snapshot(records[0], path)
    return records[0]


def _bounded_labels(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read exactly the manifest-bound prefix without probing the next byte."""

    with path.open("rb", buffering=0) as stream:
        payload = stream.read(LABEL_PREFIX_BYTES)
        returned_position = stream.tell()
    digest = _sha256_bytes(payload)
    if returned_position != LABEL_PREFIX_BYTES:
        raise AssertionError("bounded label reader returned a short prefix")
    if digest != LABEL_PREFIX_SHA256:
        raise AssertionError("bounded label prefix hash changed")
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    expected_columns = ("kst_dtm", *TARGET_COLS)
    if len(frame) != LABEL_PREFIX_ROWS or tuple(frame.columns) != expected_columns:
        raise AssertionError("bounded label prefix schema changed")
    index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("kst_dtm"), errors="raise"),
        name="forecast_kst_dtm",
    )
    if index.max() != LABEL_PREFIX_LAST_TIMESTAMP or index.has_duplicates:
        raise AssertionError("bounded label prefix time scope changed")
    frame.index = index
    frame = frame.astype(np.float64)
    return frame, {
        "path": str(path.resolve()),
        "physical_byte_limit": LABEL_PREFIX_BYTES,
        "bytes_returned_to_parser": len(payload),
        "stream_position_after_single_read": returned_position,
        "sha256": digest,
        "data_rows": len(frame),
        "last_timestamp": index.max().isoformat(),
        "full_file_hash_computed": False,
        "next_byte_probed": False,
        "future_label_bytes_read": 0,
        "2024_label_value_cells_materialized": 0,
        "2025_label_value_cells_materialized": 0,
    }


def _row_key_payload(oof: pd.DataFrame) -> bytes:
    lines: list[str] = []
    for timestamp, group, fold in oof.loc[
        :, ["forecast_kst_dtm", "group", "fold"]
    ].itertuples(index=False, name=None):
        stamp = pd.Timestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%S")
        lines.append(f"{stamp}|{group}|{fold}\n")
    return "".join(lines).encode("utf-8")


def _score_group(
    actual: np.ndarray,
    prediction_cf: np.ndarray,
    group: str,
) -> dict[str, Any]:
    capacity = float(CAPACITY_KWH[group])
    metric = group_metrics(
        actual,
        prediction_cf * capacity,
        capacity,
        group_name=group,
    )
    return {
        "score": 0.5 * (metric.one_minus_nmae + metric.ficr),
        "one_minus_nmae": metric.one_minus_nmae,
        "ficr": metric.ficr,
        "n_evaluated": metric.n_evaluated,
        "actual_energy_kwh": metric.actual_energy_kwh,
    }


def _component_delta(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    delta_n = float(
        candidate["one_minus_nmae"] - control["one_minus_nmae"]
    )
    delta_f = float(candidate["ficr"] - control["ficr"])
    return {
        "score": float(0.5 * (delta_n + delta_f)),
        "one_minus_nmae": delta_n,
        "ficr": delta_f,
    }


def _macro_delta(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    result = _component_delta(control, candidate)
    result["score"] = float(candidate["score"] - control["score"])
    return result


def _macro_scores(
    oof: pd.DataFrame,
    labels: pd.DataFrame,
    prediction_column: str,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for group in TARGET_COLS:
        rows = oof.loc[oof["group"] == group]
        times = pd.DatetimeIndex(rows["forecast_kst_dtm"])
        actual = labels.loc[times, group].to_numpy(dtype=np.float64)
        prediction = rows[prediction_column].to_numpy(dtype=np.float64)
        groups[group] = _score_group(actual, prediction, group)
    return {
        "score": float(np.mean([value["score"] for value in groups.values()])),
        "one_minus_nmae": float(
            np.mean([value["one_minus_nmae"] for value in groups.values()])
        ),
        "ficr": float(np.mean([value["ficr"] for value in groups.values()])),
        "by_group": groups,
    }


def _verify_direct_score_lock(
    direct_scores: Mapping[str, Any],
    control_macro: Mapping[str, Any],
    exit_macro: Mapping[str, Any],
) -> None:
    locked_control = direct_scores["control_direct_macro"]
    locked_exit = direct_scores["variants"][EXIT]["direct_macro"]
    checks = (
        (control_macro["score"], locked_control["total_score"]),
        (control_macro["one_minus_nmae"], locked_control["one_minus_nmae"]),
        (control_macro["ficr"], locked_control["ficr"]),
        (exit_macro["score"], locked_exit["total_score"]),
        (exit_macro["one_minus_nmae"], locked_exit["one_minus_nmae"]),
        (exit_macro["ficr"], locked_exit["ficr"]),
    )
    if any(float(observed) != float(expected) for observed, expected in checks):
        raise AssertionError("bounded exact metric no longer equals locked scores")


def _cell_evidence(
    oof: pd.DataFrame,
    labels: pd.DataFrame,
    direct_scores: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key, locked in direct_scores["registered_cells"].items():
        group, fold = key.split("::", 1)
        rows = oof.loc[(oof["group"] == group) & (oof["fold"] == fold)]
        times = pd.DatetimeIndex(rows["forecast_kst_dtm"])
        actual = labels.loc[times, group].to_numpy(dtype=np.float64)
        control = _score_group(
            actual,
            rows[CONTROL].to_numpy(dtype=np.float64),
            group,
        )
        direct = _score_group(
            actual,
            rows[EXIT].to_numpy(dtype=np.float64),
            group,
        )
        cap = _score_group(
            actual,
            rows["nonwind_cap"].to_numpy(dtype=np.float64),
            group,
        )
        direct_delta = _component_delta(control, direct)
        locked_delta = locked["candidates"][EXIT]["delta"]
        if (
            direct_delta["score"] != float(locked_delta["total"])
            or direct_delta["one_minus_nmae"]
            != float(locked_delta["one_minus_nmae"])
            or direct_delta["ficr"] != float(locked_delta["ficr"])
        ):
            raise AssertionError(f"direct cell score lock changed: {key}")
        records.append(
            {
                "cell": key,
                "group": group,
                "fold": fold,
                "rows": len(rows),
                "control": control,
                "direct_nonwind_exit": direct,
                "cap_candidate": cap,
                "direct_delta": direct_delta,
                "cap_delta": _component_delta(control, cap),
            }
        )
    if len(records) != 10:
        raise AssertionError("registered cell count changed")
    return records


def _hash_code_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/run_virtual_feature_exit.py",
        ROOT / "src/metric.py",
    )
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    }


def main() -> int:
    args = parse_args()
    experiment_root = args.experiment_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() or output.parent.exists():
        raise FileExistsError(
            f"unique no-overwrite audit path must be absent: {output.parent}"
        )

    inner_manifest_path = experiment_root / "inner/manifest.json"
    inner_oof_path = experiment_root / "inner/inner_oof_predictions.parquet"
    inner_scores_path = experiment_root / "inner/inner_direct_scores.json"
    stress_manifest_path = experiment_root / "stress/manifest.json"
    stress_ledger_path = (
        experiment_root / "stress/bootstrap_draw_and_candidate_ledger.json"
    )
    bootstrap_path = experiment_root / "stress/bootstrap_draws.npz"
    preregister_copy = experiment_root / "preregister.json"
    for path in (
        inner_manifest_path,
        inner_oof_path,
        inner_scores_path,
        stress_manifest_path,
        stress_ledger_path,
        bootstrap_path,
        preregister_copy,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(preregister_copy) != PREREGISTER_SHA256:
        raise AssertionError("frozen preregister copy changed")

    inner_manifest = json.loads(inner_manifest_path.read_text(encoding="utf-8"))
    if inner_manifest.get("preregister_sha256") != PREREGISTER_SHA256:
        raise AssertionError("INNER manifest preregister binding changed")
    oof_snapshot = _manifest_snapshot(inner_manifest, inner_oof_path)
    score_snapshot = _manifest_snapshot(inner_manifest, inner_scores_path)
    direct_scores = json.loads(inner_scores_path.read_text(encoding="utf-8"))

    stress_manifest = json.loads(stress_manifest_path.read_text(encoding="utf-8"))
    bootstrap_snapshot = _manifest_snapshot(stress_manifest, bootstrap_path)
    ledger_snapshot = _manifest_snapshot(stress_manifest, stress_ledger_path)
    stress_ledger = json.loads(stress_ledger_path.read_text(encoding="utf-8"))
    if stress_ledger.get("preregister_sha256") != PREREGISTER_SHA256:
        raise AssertionError("stress ledger preregister binding changed")

    oof = pd.read_parquet(inner_oof_path, engine="pyarrow")
    expected_columns = ("forecast_kst_dtm", "group", "fold", *VARIANTS)
    if tuple(oof.columns) != expected_columns or len(oof) != 16_128:
        raise AssertionError("INNER OOF schema or row count changed")
    oof["forecast_kst_dtm"] = pd.to_datetime(
        oof["forecast_kst_dtm"], errors="raise"
    )
    if oof.duplicated(["group", "fold", "forecast_kst_dtm"]).any():
        raise AssertionError("INNER OOF row keys are duplicated")
    prediction_values = oof.loc[:, list(VARIANTS)].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction_values).all():
        raise AssertionError("INNER OOF predictions contain non-finite values")

    control = oof[CONTROL].to_numpy(dtype=np.float64)
    exited = oof[EXIT].to_numpy(dtype=np.float64)
    increment = exited - control
    absolute_increment = np.abs(increment)
    sorted_absolute = np.sort(absolute_increment, kind="stable")
    lower_middle_index = len(sorted_absolute) // 2 - 1
    upper_middle_index = len(sorted_absolute) // 2
    lower_middle = float(sorted_absolute[lower_middle_index])
    upper_middle = float(sorted_absolute[upper_middle_index])
    cap_cf = float(np.median(absolute_increment))
    if cap_cf != EXPECTED_CAP_CF:
        raise AssertionError(f"cap derivation changed: {cap_cf!r}")
    if cap_cf != 0.5 * (lower_middle + upper_middle):
        raise AssertionError("even-row median identity changed")
    capped = np.clip(
        control + np.clip(increment, -cap_cf, cap_cf),
        0.0,
        1.02,
    )
    oof["nonwind_cap"] = capped

    labels, label_access = _bounded_labels(args.label_file.expanduser().resolve())
    control_macro = _macro_scores(oof, labels, CONTROL)
    exit_macro = _macro_scores(oof, labels, EXIT)
    cap_macro = _macro_scores(oof, labels, "nonwind_cap")
    _verify_direct_score_lock(direct_scores, control_macro, exit_macro)
    cap_macro_delta = _macro_delta(control_macro, cap_macro)
    cells = _cell_evidence(oof, labels, direct_scores)
    cell_deltas = [float(record["cap_delta"]["score"]) for record in cells]
    positive_cell_count = int(sum(value > 0.0 for value in cell_deltas))
    worst_cell = min(cells, key=lambda record: record["cap_delta"]["score"])

    stored_npz = np.load(bootstrap_path, allow_pickle=False)
    stored_order = tuple(str(value) for value in stored_npz["candidate_order"])
    stored_draws = np.asarray(stored_npz["delta_score"], dtype=np.float64)
    if stored_order != EXIT_VARIANTS:
        raise AssertionError("stored bootstrap candidate order changed")
    recomputed_original = _bootstrap_inner_deltas(oof, labels)
    if not np.array_equal(recomputed_original, stored_draws):
        raise AssertionError("registered bootstrap draws are not bit reproducible")

    cap_probe = oof.loc[:, list(expected_columns)].copy()
    for variant in EXIT_VARIANTS:
        cap_probe[variant] = capped
    cap_draw_matrix = _bootstrap_inner_deltas(cap_probe, labels)
    if not all(
        np.array_equal(cap_draw_matrix[:, 0], cap_draw_matrix[:, column])
        for column in range(1, cap_draw_matrix.shape[1])
    ):
        raise AssertionError("duplicated cap bootstrap columns diverged")
    cap_draws = cap_draw_matrix[:, 0]

    observed_original = np.asarray(
        [
            float(direct_scores["variants"][variant]["delta_score"])
            for variant in EXIT_VARIANTS
        ],
        dtype=np.float64,
    )
    observed_five = np.concatenate(
        [observed_original, [float(cap_macro_delta["score"])]]
    )
    draws_five = np.column_stack([stored_draws, cap_draws])
    maximum_centered = np.max(draws_five - observed_five[None, :], axis=1)
    c90 = float(np.quantile(maximum_centered, MAX_T_QUANTILE))
    simultaneous_lcb = float(cap_macro_delta["score"] - c90)
    minimum_cell_delta = float(min(cell_deltas))
    penalty = float(
        max(0.0, -minimum_cell_delta)
        + 0.5 * max(0.0, -cap_macro_delta["ficr"])
        + 0.5 * max(0.0, -(cap_macro_delta["one_minus_nmae"] + 0.0002))
    )
    risk_adjusted_increment = float(simultaneous_lcb - penalty)
    bootstrap_summary = {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "block_hours": BLOCK_HOURS,
        "block_description": (
            "7-operating-day circular blocks resampled within fold; common RNG "
            "draws across candidates and overlapping G1/G2 groups"
        ),
        "stored_original_draws_bit_reproduced": True,
        "stored_original_draws_sha256": _sha256_bytes(
            _float64_payload(stored_draws)
        ),
        "cap_draws_sha256": _sha256_bytes(_float64_payload(cap_draws)),
        "mean_delta": float(np.mean(cap_draws)),
        "std_delta_ddof1": float(np.std(cap_draws, ddof=1)),
        "probability_delta_positive": float(np.mean(cap_draws > 0.0)),
        "percentile_05": float(np.quantile(cap_draws, 0.05)),
        "percentile_95": float(np.quantile(cap_draws, 0.95)),
    }

    row_key_bytes = _row_key_payload(oof)
    evidence: dict[str, Any] = {
        "schema_version": "nonwind_cap_v3_evidence_v1",
        "status": "posthoc_evidence_only_not_promoted_not_submission_authority",
        "intended_use": (
            "immutable evidence input for a new, separately frozen v3 preregister"
        ),
        "formula": {
            "control_candidate": CONTROL,
            "exit_candidate": EXIT,
            "units": "capacity fraction",
            "cap_derivation": (
                "ordinary NumPy median of abs(nonwind_atmospheric-control_all) "
                "over every persisted INNER OOF row in exact Parquet row order"
            ),
            "cap_cf": _float_evidence(cap_cf),
            "expression": (
                "clip(control_all + clip(nonwind_atmospheric-control_all, "
                "-cap_cf, +cap_cf), 0.0, 1.02)"
            ),
            "inference_inputs": "two model predictions only; no target/group/month selector",
            "weight_or_threshold_search_count": 0,
        },
        "cap_derivation_provenance": {
            "row_count": len(oof),
            "row_order": "exact persisted inner_oof_predictions.parquet order",
            "row_key_encoding": "UTF-8 forecast_KST_ISO_seconds|group|fold\\n",
            "row_key_order_sha256": _sha256_bytes(row_key_bytes),
            "first_row_key": row_key_bytes.splitlines()[0].decode("utf-8"),
            "last_row_key": row_key_bytes.splitlines()[-1].decode("utf-8"),
            "group_fold_row_counts": {
                f"{group}::{fold}": int(len(rows))
                for (group, fold), rows in oof.groupby(
                    ["group", "fold"], sort=False
                )
            },
            "control_float64_row_order_sha256": _sha256_bytes(
                _float64_payload(control)
            ),
            "exit_float64_row_order_sha256": _sha256_bytes(
                _float64_payload(exited)
            ),
            "signed_increment_float64_row_order_sha256": _sha256_bytes(
                _float64_payload(increment)
            ),
            "absolute_increment_float64_row_order_sha256": _sha256_bytes(
                _float64_payload(absolute_increment)
            ),
            "sorted_absolute_increment_float64_sha256": _sha256_bytes(
                _float64_payload(sorted_absolute)
            ),
            "even_median_lower_index_zero_based": lower_middle_index,
            "even_median_upper_index_zero_based": upper_middle_index,
            "even_median_lower": _float_evidence(lower_middle),
            "even_median_upper": _float_evidence(upper_middle),
            "derived_cap": _float_evidence(cap_cf),
            "capped_candidate_float64_row_order_sha256": _sha256_bytes(
                _float64_payload(capped)
            ),
        },
        "exact_rescore": {
            "metric": "src.metric.group_metrics exact official group formula",
            "control_macro": control_macro,
            "direct_nonwind_exit_macro": exit_macro,
            "cap_candidate_macro": cap_macro,
            "direct_nonwind_exit_macro_delta": _macro_delta(
                control_macro, exit_macro
            ),
            "cap_candidate_macro_delta": cap_macro_delta,
            "cells": cells,
            "positive_cell_count": positive_cell_count,
            "all_ten_cells_positive": positive_cell_count == 10,
            "worst_cell": worst_cell["cell"],
            "worst_cell_delta": worst_cell["cap_delta"],
        },
        "simultaneous_virtual_score_evidence": {
            "ever_tried_candidate_order": [*EXIT_VARIANTS, "nonwind_cap"],
            "observed_score_deltas": observed_five.tolist(),
            "max_t_quantile_probability": MAX_T_QUANTILE,
            "five_candidate_max_t_c90": c90,
            "simultaneous_lcb": simultaneous_lcb,
            "risk_penalty_formula": (
                "max(0,-worst_cell)+0.5*max(0,-delta_F)+"
                "0.5*max(0,-(delta_N+0.0002))"
            ),
            "minimum_cell_delta": minimum_cell_delta,
            "delta_F": cap_macro_delta["ficr"],
            "delta_N": cap_macro_delta["one_minus_nmae"],
            "risk_penalty": penalty,
            "risk_adjusted_increment_R": risk_adjusted_increment,
            "frozen_R_gate": 0.003,
            "passes_frozen_R_gate": risk_adjusted_increment >= 0.003,
            "other_gate_checks": {
                "delta_F_ge_0.005": cap_macro_delta["ficr"] >= 0.005,
                "delta_N_ge_minus_0.0002": (
                    cap_macro_delta["one_minus_nmae"] >= -0.0002
                ),
                "at_least_8_of_10_cells_positive": positive_cell_count >= 8,
                "worst_cell_ge_minus_0.001": minimum_cell_delta >= -0.001,
            },
            "promotion_conclusion": (
                "not safe for current promotion: R is below the frozen 0.003 gate"
            ),
        },
        "bootstrap": bootstrap_summary,
        "deployment_order_if_a_future_v3_passes": {
            "step_1": (
                "apply the cap to the lgb_q07 component CF increment before assembly"
            ),
            "step_2": (
                "replace only lgb_q07 in the immutable locked-v3 component assembly; "
                "preserve weights, affine terms, clipping, and G2 power-bin logic"
            ),
            "step_3_champion_overlay": (
                "none by default; any scale097/rescue/champion overlay requires an "
                "independent immutable preregister and causal gate"
            ),
            "current_test_application_authorized": False,
        },
        "input_locks": {
            "preregister_copy": describe_file(preregister_copy),
            "inner_manifest": describe_file(inner_manifest_path),
            "inner_oof": oof_snapshot,
            "inner_direct_scores": score_snapshot,
            "stress_manifest": describe_file(stress_manifest_path),
            "stress_ledger": ledger_snapshot,
            "bootstrap_draws": bootstrap_snapshot,
            "label_prefix": label_access,
        },
        "access_ledger": {
            "inner_oof_arrays_read": 1,
            "inner_score_json_read": 1,
            "registered_bootstrap_npz_read": 1,
            "bounded_train_label_prefix_reads": 1,
            "bounded_train_label_bytes": LABEL_PREFIX_BYTES,
            "2024_label_bytes_read": 0,
            "2024_label_value_cells_materialized": 0,
            "2025_label_bytes_read": 0,
            "2025_label_value_cells_materialized": 0,
            "test_array_files_opened": 0,
            "weather_array_files_opened": 0,
            "submission_csv_files_opened": 0,
            "csv_outputs_written": 0,
            "model_fits": 0,
            "model_predictions_newly_generated": 0,
        },
        "reproduction_code_identity": {
            "files": _hash_code_identity(),
            "bootstrap_function": (
                "scripts.run_virtual_feature_exit._bootstrap_inner_deltas"
            ),
            "metric_function": "src.metric.group_metrics",
            "original_registered_draws_bit_reproduced": True,
            "deterministic_json_sort_keys": True,
        },
        "no_overwrite": {
            "preflight_unique_parent_absent": True,
            "output_path": str(output),
            "overwrite_allowed": False,
        },
        "leaderboard_score_claimed": False,
        "public_or_private_score_estimate": False,
    }

    output.parent.mkdir(parents=True, exist_ok=False)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _json_ready(evidence),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        if output.exists():
            raise FileExistsError(output)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        json.dumps(
            {
                "output": str(output),
                "size_bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "cap_cf": cap_cf,
                "risk_adjusted_increment_R": risk_adjusted_increment,
                "passes_frozen_R_gate": risk_adjusted_increment >= 0.003,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

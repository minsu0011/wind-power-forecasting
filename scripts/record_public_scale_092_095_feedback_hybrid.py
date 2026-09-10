"""Record four future Public triplets and build one non-submitting groupwise hybrid."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.manifest import sha256_file, utc_now, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/public_scale_092_095_feedback_hybrid_preregister_v1.json"
CONFIG_SHA256 = "aaf9d70af58c729ffd2d15c4143a51a8b4007683d3c420d2aa82e53e3691d00f"
DEFAULT_FEEDBACK = PROJECT_ROOT / "artifacts/feedback/public_scale_092_095_g1_g3_after_midnight.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts/postgate/public_scale_092_095_groupwise_hybrid_feedback_v1"
COMPONENTS = ("score", "one_minus_nmae", "ficr")
FACTORS = ("0.95", "0.92")
OBSERVED_GROUPS = ("kpx_group_1", "kpx_group_3")
G2 = "kpx_group_2"
SCHEMA = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
IDENTITY_TOLERANCE = Decimal("0.0000000005")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Test-only mode; feedback and output must both be below the OS temp directory.",
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _load_config() -> dict[str, Any]:
    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise AssertionError("preregister hash differs")
    config = _read_json(CONFIG_PATH)
    state = config["pre_feedback_state"]
    if state != {
        "future_group_only_metric_triplets_materialized": 0,
        "selection_values_materialized": 0,
        "selected_groups_or_factors": [],
        "hybrid_csv_bytes": 0,
        "actual_feedback_file_exists": False,
        "actual_output_directory_exists": False,
    }:
        raise AssertionError("preregister pre-feedback zero state differs")
    return config


def _resolved(spec: Mapping[str, Any]) -> Path:
    path = Path(str(spec["path"]))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(spec["size_bytes"]):
        raise AssertionError(f"size differs: {path}")
    if sha256_file(path) != str(spec["sha256"]).lower():
        raise AssertionError(f"hash differs: {path}")
    return path


def _decimal(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0 or result > 1:
        raise ValueError(f"{field} must be finite within [0,1]")
    return result


def validate_metric_triplet(raw: Mapping[str, Any], *, label: str) -> dict[str, Decimal]:
    values = {name: _decimal(raw[name], field=f"{label}.{name}") for name in COMPONENTS}
    expected = (values["one_minus_nmae"] + values["ficr"]) / Decimal(2)
    if abs(values["score"] - expected) > IDENTITY_TOLERANCE:
        raise ValueError(f"{label}: score identity differs")
    return values


def _read_text_csv(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    if not raw.startswith(b"\xef\xbb\xbf"):
        raise AssertionError(f"CSV BOM differs: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    if tuple(frame.columns) != SCHEMA or len(frame) != 8760:
        raise AssertionError(f"CSV schema/rows differ: {path}")
    return frame


def load_canonical_texts(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]], dict[str, Path]]:
    specs = config["canonical_csvs"]
    base_path = _resolved(specs["base"])
    base = _read_text_csv(base_path)
    paths: dict[str, Path] = {"base": base_path}
    global_frames: dict[str, pd.DataFrame] = {}
    for factor in FACTORS:
        key = f"global_{factor}"
        path = _resolved(specs[key])
        paths[key] = path
        frame = _read_text_csv(path)
        if not frame.loc[:, list(SCHEMA[:2])].equals(base.loc[:, list(SCHEMA[:2])]):
            raise AssertionError(f"global {factor} ids/timestamps differ")
        global_frames[factor] = frame
    group_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for factor in FACTORS:
        group_frames[factor] = {}
        for group in TARGET_COLS:
            spec = specs["group_factor"][factor][group]
            path = _resolved(spec)
            paths[f"{factor}:{group}"] = path
            frame = _read_text_csv(path)
            if not frame.loc[:, list(SCHEMA[:2])].equals(base.loc[:, list(SCHEMA[:2])]):
                raise AssertionError(f"{factor}:{group} ids/timestamps differ")
            if not frame[group].equals(global_frames[factor][group]):
                raise AssertionError(f"{factor}:{group} target text differs from global")
            for other in TARGET_COLS:
                if other != group and not frame[other].equals(base[other]):
                    raise AssertionError(f"{factor}:{group} non-target text differs: {other}")
                if other != group and not np.array_equal(
                    frame[other].astype(np.float64).to_numpy().view(np.uint64),
                    base[other].astype(np.float64).to_numpy().view(np.uint64),
                ):
                    raise AssertionError(f"{factor}:{group} non-target float bits differ: {other}")
            group_frames[factor][group] = frame
    return base, group_frames, paths


def validate_feedback(
    payload: Mapping[str, Any], config: Mapping[str, Any], canonical_paths: Mapping[str, Path]
) -> dict[str, dict[str, dict[str, str]]]:
    if payload.get("schema_version") != 1 or not isinstance(payload.get("group_only_results"), list):
        raise ValueError("feedback schema differs")
    rows = payload["group_only_results"]
    if len(rows) != 4:
        raise ValueError("exactly four group-only results are required")
    required_pairs = {(factor, group) for factor in FACTORS for group in OBSERVED_GROUPS}
    expected_fields = {"factor", "group", "file", "sha256", *COMPONENTS}
    observed: dict[str, dict[str, dict[str, str]]] = {factor: {} for factor in FACTORS}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("feedback row fields differ")
        pair = (str(row["factor"]), str(row["group"]))
        if pair not in required_pairs or pair in seen:
            raise ValueError("feedback factor/group pair differs or repeats")
        seen.add(pair)
        factor, group = pair
        path = Path(str(row["file"]))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        expected_path = canonical_paths[f"{factor}:{group}"]
        expected_sha = config["canonical_csvs"]["group_factor"][factor][group]["sha256"]
        if path != expected_path or str(row["sha256"]).lower() != expected_sha:
            raise ValueError(f"feedback artifact identity differs: {factor}:{group}")
        if sha256_file(path) != expected_sha:
            raise AssertionError(f"feedback artifact bytes changed: {factor}:{group}")
        metrics = validate_metric_triplet(row, label=f"{factor}:{group}")
        observed[factor][group] = {
            **{name: format(metrics[name], "f") for name in COMPONENTS},
            "file": str(path),
            "sha256": expected_sha,
        }
    if seen != required_pairs:
        raise AssertionError("required feedback pair set differs")
    return observed


def infer_and_select(
    observed: Mapping[str, Mapping[str, Mapping[str, str]]], config: Mapping[str, Any]
) -> dict[str, Any]:
    base = {
        name: Decimal(config["metric_contract"]["base"][name]) for name in COMPONENTS
    }
    deltas: dict[str, dict[str, dict[str, Decimal]]] = {factor: {} for factor in FACTORS}
    for factor in FACTORS:
        global_values = {
            name: Decimal(config["metric_contract"]["global_by_factor"][factor][name])
            for name in COMPONENTS
        }
        for group in OBSERVED_GROUPS:
            deltas[factor][group] = {
                name: Decimal(observed[factor][group][name]) - base[name]
                for name in COMPONENTS
            }
        deltas[factor][G2] = {
            name: global_values[name]
            - base[name]
            - deltas[factor]["kpx_group_1"][name]
            - deltas[factor]["kpx_group_3"][name]
            for name in COMPONENTS
        }
        for name in COMPONENTS:
            total = sum((deltas[factor][group][name] for group in TARGET_COLS), Decimal(0))
            if total != global_values[name] - base[name]:
                raise AssertionError(f"Decimal additivity differs: {factor}:{name}")
    selected: dict[str, str] = {}
    for group in TARGET_COLS:
        choices = [
            ("identity", Decimal(0)),
            ("0.95", deltas["0.95"][group]["score"]),
            ("0.92", deltas["0.92"][group]["score"]),
        ]
        # max returns the first item on exact ties, preserving identity > .95 > .92.
        selected[group] = max(choices, key=lambda item: item[1])[0]
    return {
        "decimal_additivity_exact": True,
        "group_factor_deltas": {
            factor: {
                group: {
                    "source": "inferred_by_decimal_macro_additivity" if group == G2 else "observed_group_only_minus_base",
                    **{name: format(deltas[factor][group][name], "f") for name in COMPONENTS},
                }
                for group in TARGET_COLS
            }
            for factor in FACTORS
        },
        "selected_factor_by_group": selected,
        "tie_order": ["identity", "0.95", "0.92"],
        "public_adaptive": True,
        "selection_unsafe": True,
        "private_risk": "HIGH",
    }


def compose_hybrid(
    base: pd.DataFrame,
    group_frames: Mapping[str, Mapping[str, pd.DataFrame]],
    selection: Mapping[str, str],
) -> pd.DataFrame:
    hybrid = base.copy(deep=True)
    for group in TARGET_COLS:
        factor = selection[group]
        source = base if factor == "identity" else group_frames[factor][group]
        hybrid[group] = source[group]
        if not hybrid[group].equals(source[group]):
            raise AssertionError(f"selected target text differs: {group}")
        if not np.array_equal(
            hybrid[group].astype(np.float64).to_numpy().view(np.uint64),
            source[group].astype(np.float64).to_numpy().view(np.uint64),
        ):
            raise AssertionError(f"selected target float bits differ: {group}")
    if not hybrid.loc[:, list(SCHEMA[:2])].equals(base.loc[:, list(SCHEMA[:2])]):
        raise AssertionError("hybrid ids/timestamps differ")
    values = hybrid.loc[:, list(TARGET_COLS)].astype(np.float64)
    for group in TARGET_COLS:
        array = values[group].to_numpy()
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1.02 * CAPACITY_KWH[group] + 5e-7:
            raise AssertionError(f"hybrid finite/bounds differ: {group}")
    return hybrid


def _record(path: Path, final_path: Path) -> dict[str, Any]:
    return {
        "path": str(final_path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _assert_synthetic_paths(feedback: Path, out_dir: Path) -> None:
    temp_root = Path(tempfile.gettempdir()).resolve()
    if not feedback.is_relative_to(temp_root) or not out_dir.is_relative_to(temp_root):
        raise ValueError("--synthetic paths must be below the OS temp directory")
    if feedback == DEFAULT_FEEDBACK.resolve() or out_dir == DEFAULT_OUT_DIR.resolve():
        raise ValueError("--synthetic may not use actual feedback/output paths")


def run(feedback_path: Path, out_dir: Path, *, synthetic: bool = False) -> dict[str, Any]:
    feedback_path = feedback_path.resolve()
    out_dir = out_dir.resolve()
    if synthetic:
        _assert_synthetic_paths(feedback_path, out_dir)
    if out_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {out_dir}")
    if not feedback_path.is_file():
        raise FileNotFoundError(feedback_path)
    config = _load_config()
    base, group_frames, canonical_paths = load_canonical_texts(config)
    feedback = _read_json(feedback_path)
    observed = validate_feedback(feedback, config, canonical_paths)
    # This is the first selection call: all four physical identities and triplets are now validated.
    decision = infer_and_select(observed, config)
    hybrid = compose_hybrid(base, group_frames, decision["selected_factor_by_group"])

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = out_dir.with_name(f".{out_dir.name}.staging-{os.getpid()}")
    quarantine = out_dir.with_name(f"_failed_{out_dir.name}_{os.getpid()}")
    if stage.exists() or quarantine.exists():
        raise FileExistsError("staging/quarantine path already exists")
    stage.mkdir()
    try:
        feedback_copy = stage / "feedback_input.json"
        shutil.copyfile(feedback_path, feedback_copy)
        csv_name = config["output_contract"]["hybrid_csv_name"]
        csv_path = stage / csv_name
        hybrid.to_csv(csv_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        replay = _read_text_csv(csv_path)
        if not replay.equals(hybrid):
            raise AssertionError("hybrid CSV text roundtrip differs")
        hybrid_sha = sha256_file(csv_path)
        forbidden = set(config["duplicate_guard"]["known_sha256"])
        if hybrid_sha in forbidden:
            raise ValueError(f"duplicate submission SHA forbidden: {hybrid_sha}")

        decision_path = stage / config["output_contract"]["decision_name"]
        decision_payload = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA256,
            "validated_feedback": observed,
            "decision": decision,
            "hybrid": {
                "csv": _record(csv_path, out_dir / csv_name),
                "target_text_and_float64_bits_exact_to_selected_canonical_sources": True,
                "id_timestamp_text_exact_to_base": True,
                "utf8_sig_bom_schema_8760_finite_bounds": True,
                "duplicate_known_submission_sha": False,
            },
            "public_adaptive": True,
            "selection_unsafe": True,
            "private_risk": "HIGH",
            "private_champion": False,
            "submission_performed": False,
            "upload_capability": False,
        }
        write_json_atomic(decision_path, decision_payload, overwrite=False)
        manifest_path = stage / config["output_contract"]["manifest_name"]
        input_paths = [CONFIG_PATH, feedback_path, *canonical_paths.values()]
        manifest = {
            "schema_version": 1,
            "artifact_type": "public_scale_092_095_feedback_groupwise_hybrid_v1",
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA256,
            "synthetic": synthetic,
            "inputs": [
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in input_paths
            ],
            "outputs": [
                _record(feedback_copy, out_dir / "feedback_input.json"),
                _record(decision_path, out_dir / decision_path.name),
                _record(csv_path, out_dir / csv_name),
            ],
            "selection_after_all_four_feedback_rows_only": True,
            "decimal_additivity_exact": True,
            "canonical_target_text_only": True,
            "duplicate_guard_passed": True,
            "public_adaptive": True,
            "selection_unsafe": True,
            "private_risk": "HIGH",
            "private_champion": False,
            "submission_performed": False,
            "upload_capability": False,
        }
        write_json_atomic(manifest_path, manifest, overwrite=False)
        for record in manifest["inputs"]:
            path = Path(record["path"])
            if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
                raise AssertionError(f"input mutated during recorder run: {path}")
        os.replace(stage, out_dir)
    except Exception:
        if stage.exists():
            os.replace(stage, quarantine)
        raise
    return {
        "out_dir": str(out_dir),
        "hybrid_csv": str(out_dir / config["output_contract"]["hybrid_csv_name"]),
        "selected_factor_by_group": decision["selected_factor_by_group"],
        "selection_unsafe": True,
        "private_risk": "HIGH",
        "submission_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args.feedback, args.out_dir, synthetic=args.synthetic)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

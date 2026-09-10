"""Record two future group-only Public results and build a non-submitting hybrid."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.manifest import describe_file, sha256_file, utc_now, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/public_group_scale095_probe_20260808.json"
CONFIG_SHA256 = "2ef22980a0f425988bfe7acf6820a34cf19a5e2d4a1a73b048d8a9fa412f2adf"
PACK_MANIFEST = PROJECT_ROOT / "artifacts/postgate/public_group_scale095_probe/manifest.json"
COMPONENTS = ("score", "one_minus_nmae", "ficr")
BASE = {
    "score": Decimal("0.6122309211"),
    "one_minus_nmae": Decimal("0.8518461479"),
    "ficr": Decimal("0.3726156943"),
}
GLOBAL = {
    "score": Decimal("0.6128977364"),
    "one_minus_nmae": Decimal("0.8609894944"),
    "ficr": Decimal("0.3648059783"),
}
IDENTITY_TOLERANCE = Decimal("0.0000000005")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pack-manifest", type=Path, default=PACK_MANIFEST)
    return parser.parse_args(argv)


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


def infer_group_deltas(
    observed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if len(observed) != 2 or not set(observed).issubset(TARGET_COLS):
        raise ValueError("exactly two distinct canonical groups are required")
    parsed = {
        group: validate_metric_triplet(values, label=group)
        for group, values in observed.items()
    }
    missing = next(group for group in TARGET_COLS if group not in parsed)
    deltas: dict[str, dict[str, Decimal]] = {
        group: {name: values[name] - BASE[name] for name in COMPONENTS}
        for group, values in parsed.items()
    }
    global_delta = {name: GLOBAL[name] - BASE[name] for name in COMPONENTS}
    deltas[missing] = {
        name: global_delta[name] - sum(
            (deltas[group][name] for group in parsed), Decimal(0)
        )
        for name in COMPONENTS
    }
    positive = [group for group in TARGET_COLS if deltas[group]["score"] > 0]
    records = {
        group: {
            "source": "inferred_by_macro_additivity" if group == missing else "observed_group_only_minus_base",
            **{name: format(deltas[group][name], "f") for name in COMPONENTS},
        }
        for group in TARGET_COLS
    }
    for name in COMPONENTS:
        if sum((deltas[group][name] for group in TARGET_COLS), Decimal(0)) != global_delta[name]:
            raise AssertionError(f"decimal additivity differs: {name}")
    return {
        "global_delta_decimal": {name: format(global_delta[name], "f") for name in COMPONENTS},
        "group_deltas_decimal": records,
        "inferred_group": missing,
        "positive_groups_strict_total_delta": positive,
        "nonpositive_groups_identity": [group for group in TARGET_COLS if group not in positive],
        "decimal_additivity_exact": True,
        "automatic_final_recommendation": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON root must be an object")
    return value


def _verify_config() -> dict[str, Any]:
    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise AssertionError("config hash differs")
    return _read_json(CONFIG_PATH)


def _known_group_csvs(config: Mapping[str, Any], manifest_path: Path) -> dict[str, dict[str, str]]:
    manifest = _read_json(manifest_path)
    if manifest.get("config_sha256") != CONFIG_SHA256:
        raise AssertionError("pack manifest config differs")
    records = {Path(item["path"]).name: item for item in manifest["outputs"]}
    output_dir = Path(config["output_contract"]["directory"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    known: dict[str, dict[str, str]] = {}
    for group, stem in config["output_names"].items():
        name = f"{stem}.csv"
        record = records[name]
        path = (output_dir / name).resolve()
        if sha256_file(path) != record["sha256"]:
            raise AssertionError(f"canonical group CSV changed: {group}")
        known[group] = {"path": str(path), "sha256": str(record["sha256"])}
    return known


def validate_feedback(
    payload: Mapping[str, Any], known: Mapping[str, Mapping[str, str]]
) -> dict[str, dict[str, Any]]:
    if payload.get("schema_version") != 1 or not isinstance(payload.get("group_only_results"), list):
        raise ValueError("feedback schema differs")
    rows = payload["group_only_results"]
    if len(rows) != 2:
        raise ValueError("exactly two group-only results are required")
    expected_fields = {"group", "file", "sha256", *COMPONENTS}
    observed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("group result fields differ")
        group = str(row["group"])
        if group not in known or group in observed:
            raise ValueError("group must be distinct and canonical")
        path = Path(str(row["file"]))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        if str(path) != known[group]["path"] or str(row["sha256"]).lower() != known[group]["sha256"]:
            raise ValueError(f"{group}: artifact identity differs")
        if sha256_file(path) != known[group]["sha256"]:
            raise AssertionError(f"{group}: artifact bytes changed")
        metrics = validate_metric_triplet(row, label=group)
        observed[group] = {
            **{name: format(metrics[name], "f") for name in COMPONENTS},
            "file": str(path),
            "sha256": known[group]["sha256"],
        }
    return observed


def _write_csv_text(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_hybrid(
    config: Mapping[str, Any], known: Mapping[str, Mapping[str, str]], decision: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_spec = config["input_identities"]["base_csv"]
    base_path = PROJECT_ROOT / base_spec["path"]
    if sha256_file(base_path) != base_spec["sha256"]:
        raise AssertionError("base CSV changed")
    base_text = pd.read_csv(base_path, encoding="utf-8-sig", dtype="string")
    hybrid_text = base_text.copy(deep=True)
    positive = set(decision["positive_groups_strict_total_delta"])
    for group in positive:
        group_text = pd.read_csv(known[group]["path"], encoding="utf-8-sig", dtype="string")
        hybrid_text[group] = group_text[group]
    index = pd.DatetimeIndex(pd.to_datetime(hybrid_text["forecast_kst_dtm"]), name="forecast_kst_dtm")
    prediction = hybrid_text.loc[:, list(TARGET_COLS)].astype(np.float64)
    prediction.index = index
    for group in TARGET_COLS:
        values = prediction[group].to_numpy()
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1.02 * CAPACITY_KWH[group] + 5e-7:
            raise AssertionError(f"hybrid bounds/finite differs: {group}")
    return hybrid_text, prediction


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    feedback_path = args.feedback.resolve()
    out_dir = args.out_dir.resolve()
    manifest_path = args.pack_manifest.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {out_dir}")
    config = _verify_config()
    known = _known_group_csvs(config, manifest_path)
    feedback = _read_json(feedback_path)
    observed = validate_feedback(feedback, known)
    decision = infer_group_deltas(
        {group: {name: values[name] for name in COMPONENTS} for group, values in observed.items()}
    )
    out_dir.mkdir(parents=True)
    shutil.copyfile(feedback_path, out_dir / "feedback_input.json")
    hybrid_text, prediction = _build_hybrid(config, known, decision)
    parquet_path = out_dir / "positive_group_scale095_hybrid_2025.parquet"
    csv_path = out_dir / "positive_group_scale095_hybrid_2025.csv"
    prediction.to_parquet(parquet_path, engine="pyarrow", compression="zstd", index=True)
    _write_csv_text(hybrid_text, csv_path)
    replay = pd.read_csv(csv_path, encoding="utf-8-sig", dtype="string")
    if not replay.equals(hybrid_text) or not csv_path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise AssertionError("hybrid CSV text roundtrip differs")
    decision_payload = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "public_adaptive": True,
        "selection_unsafe": True,
        "private_champion": False,
        "submission_performed": False,
        "upload_capability": False,
        "final_top2_selection_included": False,
        "validated_observed_results": observed,
        "decision": decision,
        "hybrid": {
            "formula": "scale exactly the strictly positive observed/inferred groups by 0.95; identity otherwise",
            "prediction": describe_file(parquet_path),
            "csv": describe_file(csv_path),
            "automatic_final_recommendation": False,
        },
    }
    write_json_atomic(out_dir / "decision.json", decision_payload, overwrite=False)
    outputs = [out_dir / "feedback_input.json", out_dir / "decision.json", parquet_path, csv_path]
    manifest = {
        "schema_version": 1,
        "artifact_type": "public_group_scale095_feedback_and_positive_group_hybrid",
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA256,
        "public_adaptive": True,
        "selection_unsafe": True,
        "private_champion": False,
        "submission_performed": False,
        "automatic_final_recommendation": False,
        "inputs": [
            describe_file(feedback_path),
            describe_file(CONFIG_PATH),
            describe_file(manifest_path),
            describe_file(Path(__file__)),
            *[describe_file(Path(value["path"])) for value in known.values()],
        ],
        "outputs": [describe_file(path) for path in outputs],
    }
    write_json_atomic(out_dir / "manifest.json", manifest, overwrite=False)
    print(json.dumps({"out_dir": str(out_dir), "positive_groups": decision["positive_groups_strict_total_delta"], "automatic_final_recommendation": False, "submission_performed": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

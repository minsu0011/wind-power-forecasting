"""Build the frozen, non-submitting 0.96 and 0.98 global scalar bracket."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
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
from src.public_scale_probe import scale_predictions, validate_prediction_frame  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/public_global_scalar_bracket_096_098_preregister_v1.json"
CONFIG_SHA256 = "62ee197dccfd19d96e468aa94530109544746fbff762c45770c4ffa9cdbfbc5e"
CENSUS_PATH = PROJECT_ROOT / "artifacts/audits/public_global_scalar_bracket_096_098_duplicate_census_v1.json"
CENSUS_SHA256 = "b85a9d65307f6a7d4d66923f4fdd07687349d8c8e08ad16de7a35a78fb8d7a7e"
INCIDENT_PATH = PROJECT_ROOT / "artifacts/incidents/public_global_scalar_bracket_096_098_attempt1_float_rounding_assertion.json"
INCIDENT_SHA256 = "b44f5a33f674096a019e680739215ee20bee8ce588430e870d2615a497abdb0e"
EXPECTED_ROWS = 8760
SCHEMA = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
FACTORS = (0.96, 0.98)
SIX_DECIMAL = re.compile(r"^-?\d+\.\d{6}$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/postgate/public_global_scalar_bracket_096_098_v1",
    )
    return parser.parse_args(argv)


def _absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify(spec: Mapping[str, Any]) -> Path:
    path = _absolute(str(spec["path"])).resolve()
    if not path.is_file() or path.stat().st_size != int(spec["bytes"]):
        raise AssertionError(f"registered input size differs: {path}")
    if sha256_file(path) != str(spec["sha256"]):
        raise AssertionError(f"registered input SHA differs: {path}")
    return path


def verify_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    path = path.resolve()
    if path != CONFIG_PATH.resolve() or sha256_file(path) != CONFIG_SHA256:
        raise AssertionError("only the frozen scalar-bracket preregistration is allowed")
    if path.with_suffix(".sha256").read_text(encoding="utf-8") != f"{CONFIG_SHA256}  {path.name}\n":
        raise AssertionError("preregistration sidecar differs")
    if sha256_file(CENSUS_PATH) != CENSUS_SHA256:
        raise AssertionError("duplicate census differs")
    if sha256_file(INCIDENT_PATH) != INCIDENT_SHA256:
        raise AssertionError("attempt1 incident differs")
    config = json.loads(path.read_text(encoding="utf-8"))
    if tuple(float(item["factor"]) for item in config["candidates"]) != FACTORS:
        raise AssertionError("candidate factors or order differ")
    risk = config["risk"]
    if not risk["selection_unsafe"] or not risk["public_adaptive"] or risk["private_score"] != "UNKNOWN":
        raise AssertionError("risk contract differs")
    if config["later_feedback_invariance"]["adaptive_selection_or_rebuild"] is not False:
        raise AssertionError("later-feedback invariance differs")
    for spec in config["frozen_inputs"].values():
        _verify(spec)
    _verify(config["prior_public_context"]["feedback_source"])
    return config


def read_submission_text(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise AssertionError(f"BOM differs: {path}")
    text = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    if tuple(text.columns) != SCHEMA or len(text) != EXPECTED_ROWS:
        raise AssertionError(f"schema/rows differ: {path}")
    index = pd.DatetimeIndex(pd.to_datetime(text["forecast_kst_dtm"], errors="raise"), name="forecast_kst_dtm")
    prediction = text.loc[:, list(TARGET_COLS)].astype(np.float64)
    prediction.index = index
    validate_prediction_frame(prediction, name=str(path))
    return text, prediction


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(base_text: pd.DataFrame, prediction: pd.DataFrame, path: Path) -> None:
    output = base_text.loc[:, ["forecast_id", "forecast_kst_dtm"]].copy()
    for group in TARGET_COLS:
        output[group] = [f"{value:.6f}" for value in prediction[group].to_numpy(dtype=np.float64)]
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        output.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def audit_candidate(
    *, factor: float, csv_path: Path, parquet_path: Path, base_text: pd.DataFrame,
    base_prediction: pd.DataFrame, known_hashes: set[str]
) -> dict[str, Any]:
    expected = scale_predictions(base_prediction, factor)
    replay = pd.read_parquet(parquet_path, engine="pyarrow").astype(np.float64)
    replay.index = pd.DatetimeIndex(replay.index, name="forecast_kst_dtm")
    if not replay.index.equals(expected.index) or not np.array_equal(replay.to_numpy(), expected.to_numpy()):
        raise AssertionError(f"Parquet formula replay differs: {factor}")
    csv_text, csv_prediction = read_submission_text(csv_path)
    if not csv_text[["forecast_id", "forecast_kst_dtm"]].equals(base_text[["forecast_id", "forecast_kst_dtm"]]):
        raise AssertionError(f"CSV IDs/timestamps differ: {factor}")
    for group in TARGET_COLS:
        if not csv_text[group].map(lambda value: bool(SIX_DECIMAL.fullmatch(str(value)))).all():
            raise AssertionError(f"CSV six-decimal contract differs: {factor}/{group}")
        expected_text = pd.Series(
            [f"{value:.6f}" for value in expected[group].to_numpy(dtype=np.float64)],
            dtype="string",
        )
        if not csv_text[group].reset_index(drop=True).equals(expected_text):
            raise AssertionError(f"CSV raw-text formula replay differs: {factor}/{group}")
        array = csv_prediction[group].to_numpy()
        if array.min() < 0.0 or array.max() > 1.02 * CAPACITY_KWH[group] + 5e-7:
            raise AssertionError(f"CSV bounds differ: {factor}/{group}")
    csv_sha = sha256_file(csv_path)
    parquet_sha = sha256_file(parquet_path)
    if csv_sha in known_hashes or parquet_sha in known_hashes:
        raise AssertionError(f"candidate duplicates a registered scalar artifact: {factor}")
    return {
        "factor": factor,
        "csv": describe_file(csv_path),
        "parquet": describe_file(parquet_path),
        "rows_schema_bom_ids_time_finite_bounds_six_decimals": True,
        "parquet_formula_bit_exact": True,
        "csv_raw_text_exact_to_format_frozen_float64_dot6f": True,
        "csv_decimal_rounding_error_upper_bound_kwh": 0.0000005,
        "nonduplicate_against_registered_inputs": True,
    }


def _snapshot(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = list(config["frozen_inputs"].values()) + [config["prior_public_context"]["feedback_source"]]
    return [describe_file(_verify(spec)) for spec in specs]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = verify_config(args.config)
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {out_dir}")
    before = _snapshot(config)
    out_dir.mkdir(parents=True)
    base_text, base_prediction = read_submission_text(_verify(config["frozen_inputs"]["base_csv"]))
    sample_text, _ = read_submission_text(_verify(config["frozen_inputs"]["sample_csv"]))
    if not base_text[["forecast_id", "forecast_kst_dtm"]].equals(sample_text[["forecast_id", "forecast_kst_dtm"]]):
        raise AssertionError("base/sample IDs or timestamps differ")
    shutil.copyfile(CONFIG_PATH, out_dir / "preregister.json")
    shutil.copyfile(CONFIG_PATH.with_suffix(".sha256"), out_dir / "preregister.sha256")
    shutil.copyfile(CENSUS_PATH, out_dir / "duplicate_census.json")
    shutil.copyfile(INCIDENT_PATH, out_dir / "attempt1_incident.json")
    known_hashes = {str(spec["sha256"]) for spec in config["frozen_inputs"].values()}
    records: list[dict[str, Any]] = []
    for candidate in config["candidates"]:
        factor = float(candidate["factor"])
        prediction = scale_predictions(base_prediction, factor)
        csv_path = out_dir / candidate["csv"]
        parquet_path = out_dir / candidate["parquet"]
        _write_parquet(prediction, parquet_path)
        _write_csv(base_text, prediction, csv_path)
        records.append(audit_candidate(
            factor=factor, csv_path=csv_path, parquet_path=parquet_path,
            base_text=base_text, base_prediction=base_prediction, known_hashes=known_hashes,
        ))
    all_hashes = [record[k]["sha256"] for record in records for k in ("csv", "parquet")]
    if len(set(all_hashes)) != 4:
        raise AssertionError("new CSV/Parquet outputs are not four-way distinct")
    after = _snapshot(config)
    if before != after:
        raise AssertionError("protected input snapshot changed")
    results = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA256,
        "risk": config["risk"],
        "records": records,
        "all_new_outputs_four_way_distinct": True,
        "protected_inputs_nonmutation": True,
        "actual_new_feedback_records": 0,
        "selection_performed": False,
        "submission_performed": False,
    }
    write_json_atomic(out_dir / "results.json", results, overwrite=False)
    outputs = [path for path in out_dir.rglob("*") if path.is_file() and path.name not in ("manifest.json", "manifest.sha256")]
    manifest = {
        "schema_version": 1,
        "artifact_type": "public_global_scalar_bracket_096_098_v1",
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA256,
        "risk": config["risk"],
        "formula_frozen_before_new_feedback": True,
        "attempt1_incident_sha256": INCIDENT_SHA256,
        "factors": list(FACTORS),
        "protected_inputs": before,
        "outputs_excluding_manifest": [describe_file(path) for path in sorted(outputs)],
        "actual_new_feedback_records": 0,
        "selection_performed": False,
        "submission_performed": False,
    }
    write_json_atomic(out_dir / "manifest.json", manifest, overwrite=False)
    (out_dir / "manifest.sha256").write_text(
        f"{sha256_file(out_dir / 'manifest.json')}  manifest.json\n", encoding="utf-8", newline=""
    )
    print(json.dumps({"out_dir": str(out_dir), "factors": list(FACTORS), "feedback": 0, "selection": False, "submission": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

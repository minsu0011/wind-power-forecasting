"""Evaluate the frozen plain-scale097 G1/G3 plus exact v5 G2 rescue composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import scripts.run_public_adaptive_scale097_g2_delta as parent  # noqa: E402
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now  # noqa: E402


CONFIG_SHA256 = "cefac2445b383e01f140287d49eee41b84aa2f5bb53cbe7bce429f9ad8f05569"
FACTOR = np.float64(0.97)
TARGETS = parent.TARGETS
CAPACITY = parent.CAPACITY
ACTIVE_GROUP = "kpx_group_2"
IDENTITY_TO_PLAIN = ("kpx_group_1", "kpx_group_3")
SEGMENTS = parent.SEGMENTS
PROTECTED_ROOTS = (
    "artifacts/postgate/public_adaptive_scale097_g2_delta_v2",
    "artifacts/postgate/direct_interval_g2_posthoc_rescue_v5",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/public_adaptive_scale097_g1g3_g2rescue_preregister_v1.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/public_adaptive_scale097_g1g3_g2rescue_v1"),
    )
    return parser.parse_args(argv)


def _bits(series: pd.Series) -> np.ndarray:
    return np.ascontiguousarray(series.to_numpy(dtype=np.float64)).view(np.uint64)


def _verify_spec(spec: Mapping[str, Any]) -> Path:
    path = Path(str(spec["path"]))
    if not path.is_absolute():
        path = PROJECT_DIR / path
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_size = spec.get("bytes", spec.get("size_bytes"))
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise AssertionError(f"size differs: {path}")
    if sha256_file(path) != str(spec["sha256"]):
        raise AssertionError(f"hash differs: {path}")
    return path


def _verify_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if sha256_file(path) != CONFIG_SHA256:
        raise AssertionError("config hash differs")
    expected_sidecar = f"{CONFIG_SHA256}  {path.name}\n"
    if path.with_suffix(".sha256").read_text(encoding="utf-8") != expected_sidecar:
        raise AssertionError("config sidecar differs")
    config = json.loads(path.read_text(encoding="utf-8"))
    risk = config["risk_classification"]
    for name in ("public_adaptive", "posthoc_2024", "multiple_testing", "selection_unsafe"):
        if risk[name] is not True:
            raise AssertionError(f"risk disclosure differs: {name}")
    formula = config["immutable_single_composition"]
    if float(formula["factor"]) != float(FACTOR) or int(formula["candidate_count"]) != 1:
        raise AssertionError("immutable composition differs")
    _verify_spec(config["duplicate_census"])
    for spec in config["fixed_lineage"].values():
        _verify_spec(spec)
    diagnostic = config["diagnostic_2024"]
    _verify_spec(diagnostic["labels"])
    for variant in diagnostic["variants_in_fixed_order"]:
        _verify_spec(variant["baseline"])
        _verify_spec(variant["g2_rescue"])
        if "audited_plain_scale097_crosscheck" in variant:
            _verify_spec(variant["audited_plain_scale097_crosscheck"])
    final = config["final_if_and_only_if_diagnostic_passes"]
    for name in ("plain_scale097", "g2_rescue_v5", "sample"):
        _verify_spec(final[name])
    return config


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def _source_closure(args: argparse.Namespace) -> dict[str, Any]:
    files = parent.resolve_ast_closure(Path(__file__))
    test = PROJECT_DIR / "tests/test_public_adaptive_scale097_g1g3_g2rescue.py"
    return {
        "resolver": "recursive Python AST local imports plus package initializers and explicit test",
        "entrypoint": Path(__file__).resolve().relative_to(PROJECT_DIR).as_posix(),
        "resolved_relative_paths": [path.relative_to(PROJECT_DIR).as_posix() for path in files],
        "resolved_files": [describe_file(path) for path in files],
        "resolved_file_count": len(files),
        "unresolved_local_imports": [],
        "test": describe_file(test),
        "config": describe_file(args.config.resolve()),
        "sidecar": describe_file(args.config.with_suffix(".sha256").resolve()),
    }


def _protected_snapshot() -> dict[str, Any]:
    files: set[Path] = set()
    for relative in PROTECTED_ROOTS:
        root = PROJECT_DIR / relative
        if not root.is_dir():
            raise FileNotFoundError(root)
        files.update(path.resolve() for path in root.rglob("*") if path.is_file())
    records = [
        {
            "relative_path": path.relative_to(PROJECT_DIR).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]
    return {
        "schema_version": 1,
        "protected_roots": list(PROTECTED_ROOTS),
        "file_count": len(records),
        "files": records,
    }


def _read_prediction(path: Path, year: int) -> pd.DataFrame:
    index = parent._year_index(year)
    frame = pd.read_parquet(path).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index) or tuple(frame.columns) != TARGETS:
        raise AssertionError(f"prediction schema/index differs: {path}")
    if not np.isfinite(frame.to_numpy()).all():
        raise AssertionError(f"non-finite prediction: {path}")
    for group in TARGETS:
        values = frame[group].to_numpy(dtype=np.float64)
        if values.min() < 0.0 or values.max() > np.float64(1.02 * CAPACITY[group]):
            raise AssertionError(f"prediction bounds differ: {path}/{group}")
    return frame


def compose(base: pd.DataFrame, rescue: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not base.index.equals(rescue.index) or tuple(base.columns) != TARGETS or tuple(rescue.columns) != TARGETS:
        raise AssertionError("baseline/rescue alignment differs")
    for group in IDENTITY_TO_PLAIN:
        if not np.array_equal(_bits(base[group]), _bits(rescue[group])):
            raise AssertionError(f"v5 rescue identity differs: {group}")
    plain = base.copy(deep=True)
    for group in TARGETS:
        values = np.clip(
            FACTOR * base[group].to_numpy(dtype=np.float64),
            np.float64(0.0),
            np.float64(1.02 * CAPACITY[group]),
        )
        plain[group] = values
    candidate = plain.copy(deep=True)
    candidate[ACTIVE_GROUP] = rescue[ACTIVE_GROUP].to_numpy(dtype=np.float64, copy=True)
    for group in IDENTITY_TO_PLAIN:
        if not np.array_equal(_bits(candidate[group]), _bits(plain[group])):
            raise AssertionError(f"candidate/plain identity differs: {group}")
    if not np.array_equal(_bits(candidate[ACTIVE_GROUP]), _bits(rescue[ACTIVE_GROUP])):
        raise AssertionError("candidate G2 differs from exact rescue")
    return plain, candidate


def _metric_delta(
    actual: pd.DataFrame,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    group: str | None,
) -> dict[str, Any]:
    if group is None:
        reference_metric = parent._mixed(actual, reference)
        candidate_metric = parent._mixed(actual, candidate)
    else:
        reference_metric = parent._group_metrics(actual[group], reference[group], group)
        candidate_metric = parent._group_metrics(actual[group], candidate[group], group)
    return {
        "group": group if group is not None else "mixed",
        "reference": reference_metric,
        "candidate": candidate_metric,
        "delta_total_score": candidate_metric["total_score"] - reference_metric["total_score"],
        "delta_one_minus_nmae": candidate_metric["one_minus_nmae"] - reference_metric["one_minus_nmae"],
        "delta_ficr": candidate_metric["ficr"] - reference_metric["ficr"],
    }


def _evaluate_reference(
    labels: pd.DataFrame,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    groups: Sequence[str],
) -> dict[str, Any]:
    group_records: list[dict[str, Any]] = []
    mixed_records: list[dict[str, Any]] = []
    for segment, rows in parent._segments(2024).items():
        actual = labels.loc[rows, list(TARGETS)]
        for group in groups:
            record = _metric_delta(actual, reference.loc[rows], candidate.loc[rows], group)
            record["segment"] = segment
            group_records.append(record)
        mixed = _metric_delta(actual, reference.loc[rows], candidate.loc[rows], None)
        mixed["segment"] = segment
        mixed_records.append(mixed)
    full = next(record for record in mixed_records if record["segment"] == "full")
    checks = {
        "all_registered_group_total_deltas_strictly_positive": all(
            record["delta_total_score"] > 0.0 for record in group_records
        ),
        "all_mixed_total_deltas_strictly_positive": all(
            record["delta_total_score"] > 0.0 for record in mixed_records
        ),
        "full_mixed_one_minus_nmae_nonnegative": full["delta_one_minus_nmae"] >= 0.0,
        "full_mixed_ficr_nonnegative": full["delta_ficr"] >= 0.0,
    }
    return {
        "groups_required_strict": list(groups),
        "group_records": group_records,
        "mixed_records": mixed_records,
        "checks": checks,
        "passed": all(checks.values()),
        "minimum_group_delta": min(record["delta_total_score"] for record in group_records),
        "minimum_mixed_delta": min(record["delta_total_score"] for record in mixed_records),
    }


def _read_labels(spec: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(_verify_spec(spec), encoding="utf-8-sig")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm")), name="forecast_kst_dtm")
    if len(frame) != int(spec["rows"]) or tuple(frame.columns) != TARGETS:
        raise AssertionError("label schema/rows differs")
    return frame.astype(np.float64)


def _validate_csv(path: Path, sample: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    if not path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise AssertionError("CSV BOM differs")
    text = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    if tuple(text.columns) != tuple(sample.columns) or len(text) != 8760:
        raise AssertionError("CSV schema/rows differs")
    identifiers = ["forecast_id", "forecast_kst_dtm"]
    if not text[identifiers].equals(sample[identifiers].astype("string")):
        raise AssertionError("sample identifier/time differs")
    for group in TARGETS:
        expected = prediction[group].reset_index(drop=True).map(
            lambda value: f"{float(value):.6f}"
        ).astype("string")
        if not text[group].equals(expected):
            raise AssertionError(f"six-decimal roundtrip differs: {group}")
    values = text.loc[:, list(TARGETS)].astype(np.float64)
    if not np.isfinite(values.to_numpy()).all():
        raise AssertionError("CSV non-finite")
    for group in TARGETS:
        if values[group].min() < 0.0 or values[group].max() > 1.02 * CAPACITY[group] + 5e-7:
            raise AssertionError(f"CSV bounds differ: {group}")
    return {
        "BOM": True,
        "rows": 8760,
        "schema": True,
        "sample_identifier_time_order": True,
        "finite": True,
        "bounds": True,
        "six_decimal_roundtrip": True,
        "sha256": sha256_file(path),
    }


def _manifest(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    closure: Mapping[str, Any],
    before: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    final: Mapping[str, Any],
) -> None:
    if closure != _source_closure(args):
        raise AssertionError("source/config/test closure changed")
    after = _protected_snapshot()
    if before != after:
        raise AssertionError("protected upstream changed")
    after_path = args.out_dir / "protected_upstream_snapshot_after.json"
    parent._write_json(after_path, after)
    files = sorted(
        (path for path in args.out_dir.rglob("*") if path.is_file() and path.name != "manifest.json"),
        key=lambda path: path.relative_to(args.out_dir).as_posix(),
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "public_adaptive_scale097_g1g3_plus_exact_g2rescue_v1",
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA256,
        "risk": config["risk_classification"],
        "immutable_composition": config["immutable_single_composition"],
        "source_provenance": {
            "closure": closure,
            "locked_before_new_candidate_and_metric": True,
            "unchanged": True,
        },
        "protected_upstream_nonmutation": {
            "before_after_exact": True,
            "file_count": before["file_count"],
            "roots": before["protected_roots"],
        },
        "diagnostic": diagnostic,
        "final": final,
        "strict_or_private_claim": False,
        "outputs": [describe_file(path) for path in files],
        "output_count_excluding_manifest": len(files),
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
    }
    parent._write_json(args.out_dir / "manifest.json", manifest)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.config = args.config.resolve()
    args.out_dir = args.out_dir.resolve()
    config = _verify_config(args.config)
    if args.out_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {args.out_dir}")
    before = _protected_snapshot()
    args.out_dir.mkdir(parents=True)
    _copy(args.config, args.out_dir / "preregister.json")
    _copy(args.config.with_suffix(".sha256"), args.out_dir / "preregister.sha256")
    before_path = args.out_dir / "protected_upstream_snapshot_before.json"
    parent._write_json(before_path, before)
    closure = _source_closure(args)
    closure_path = args.out_dir / "source_closure_before_candidate_and_metric_lock.json"
    parent._write_json(
        closure_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA256,
            "new_candidate_cells_before_lock": 0,
            "new_metric_values_before_lock": 0,
            "new_csv_bytes_before_lock": 0,
            "closure": closure,
        },
    )

    diagnostic_config = config["diagnostic_2024"]
    variants: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    candidate_files: list[Path] = []
    for spec in diagnostic_config["variants_in_fixed_order"]:
        variant_id = str(spec["id"])
        base = _read_prediction(_verify_spec(spec["baseline"]), 2024)
        rescue = _read_prediction(_verify_spec(spec["g2_rescue"]), 2024)
        plain, candidate = compose(base, rescue)
        if "audited_plain_scale097_crosscheck" in spec:
            audited = _read_prediction(_verify_spec(spec["audited_plain_scale097_crosscheck"]), 2024)
            for group in TARGETS:
                if not np.array_equal(_bits(plain[group]), _bits(audited[group])):
                    raise AssertionError(f"audited plain crosscheck differs: {group}")
        plain_path = args.out_dir / f"diagnostic_2024/{variant_id}/plain_scale097.parquet"
        candidate_path = args.out_dir / f"diagnostic_2024/{variant_id}/candidate.parquet"
        parent._atomic_parquet(plain, plain_path)
        parent._atomic_parquet(candidate, candidate_path)
        candidate_files.extend([plain_path, candidate_path])
        variants[variant_id] = (base, plain, candidate)

    candidate_lock_path = args.out_dir / "diagnostic_candidates_before_label_lock.json"
    parent._write_json(
        candidate_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA256,
            "files": [describe_file(path) for path in candidate_files],
            "new_2024_label_cells_decoded_before_lock": 0,
            "new_2024_metric_values_before_lock": 0,
            "year_2025_or_sample_cells_before_lock": 0,
        },
    )

    labels = _read_labels(diagnostic_config["labels"])
    variant_results: dict[str, Any] = {}
    dual_pass = True
    for variant_id, (base, plain, candidate) in variants.items():
        vs_baseline = _evaluate_reference(labels, base, candidate, TARGETS)
        vs_plain = _evaluate_reference(labels, plain, candidate, (ACTIVE_GROUP,))
        identity = {
            group: bool(np.array_equal(_bits(candidate[group]), _bits(plain[group])))
            for group in IDENTITY_TO_PLAIN
        }
        passed = bool(vs_baseline["passed"] and vs_plain["passed"] and all(identity.values()))
        dual_pass = dual_pass and passed
        variant_results[variant_id] = {
            "vs_unmodified_baseline": vs_baseline,
            "vs_plain_scale097": vs_plain,
            "G1_G3_bit_identity_to_plain": identity,
            "passed": passed,
        }
    diagnostic = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "candidate_lock": describe_file(candidate_lock_path),
        "variants": variant_results,
        "dual_variant_dual_reference_passed": bool(dual_pass),
        "retuned_after_result": False,
    }
    diagnostic_path = args.out_dir / "diagnostic_results.json"
    parent._write_json(diagnostic_path, diagnostic)

    final_path = args.out_dir / "final_results.json"
    if not dual_pass:
        final = {
            "schema_version": 1,
            "performed": False,
            "reason": "dual-variant dual-reference diagnostic rejection",
            "year_2025_or_sample_cells": 0,
            "csv_created": False,
        }
        parent._write_json(final_path, final)
    else:
        final_config = config["final_if_and_only_if_diagnostic_passes"]
        final_lock_path = args.out_dir / "final_prescore_lock.json"
        parent._write_json(
            final_lock_path,
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "config_sha256": CONFIG_SHA256,
                "diagnostic": describe_file(diagnostic_path),
                "plain_scale097_identity": final_config["plain_scale097"],
                "g2_rescue_v5_identity": final_config["g2_rescue_v5"],
                "sample_identity": final_config["sample"],
                "year_2025_or_sample_cells_before_lock": 0,
            },
        )
        plain = _read_prediction(_verify_spec(final_config["plain_scale097"]), 2025)
        rescue = _read_prediction(_verify_spec(final_config["g2_rescue_v5"]), 2025)
        candidate = plain.copy(deep=True)
        candidate[ACTIVE_GROUP] = rescue[ACTIVE_GROUP].to_numpy(dtype=np.float64, copy=True)
        for group in IDENTITY_TO_PLAIN:
            if not np.array_equal(_bits(candidate[group]), _bits(plain[group])):
                raise AssertionError(f"final plain identity differs: {group}")
        if not np.array_equal(_bits(candidate[ACTIVE_GROUP]), _bits(rescue[ACTIVE_GROUP])):
            raise AssertionError("final G2 rescue identity differs")
        prediction_path = args.out_dir / final_config["prediction"]
        parent._atomic_parquet(candidate, prediction_path)
        replay = _read_prediction(prediction_path, 2025)
        for group in TARGETS:
            if not np.array_equal(_bits(replay[group]), _bits(candidate[group])):
                raise AssertionError(f"final parquet replay differs: {group}")
        sample = pd.read_csv(
            _verify_spec(final_config["sample"]),
            encoding="utf-8-sig",
            dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
        )
        if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGETS) or len(sample) != 8760:
            raise AssertionError("sample schema/rows differs")
        if not pd.DatetimeIndex(
            pd.to_datetime(sample["forecast_kst_dtm"]), name="forecast_kst_dtm"
        ).equals(candidate.index):
            raise AssertionError("sample timestamps differ")
        submission = sample.copy()
        for group in TARGETS:
            submission[group] = candidate[group].to_numpy(dtype=np.float64)
        csv_path = args.out_dir / final_config["csv"]
        parent._atomic_csv(submission, csv_path)
        validation = _validate_csv(csv_path, sample, candidate)
        final = {
            "schema_version": 1,
            "performed": True,
            "csv_created": True,
            "G1_G3_bits_exact_to_audited_plain_scale097": True,
            "G2_bits_exact_to_audited_v5_rescue": True,
            "prediction": describe_file(prediction_path),
            "csv": describe_file(csv_path),
            "validation": validation,
        }
        parent._write_json(final_path, final)

    _manifest(args, config, closure, before, diagnostic, final)
    return {
        "diagnostic_passed": bool(dual_pass),
        "csv_created": bool(final["csv_created"]),
        "manifest_sha256": sha256_file(args.out_dir / "manifest.json"),
        "out_dir": str(args.out_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(parent._json_ready(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build three immutable, non-submitting one-group-only 0.95 Public probes."""

from __future__ import annotations

import argparse
import ast
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

from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402
from src.public_group_scale_probe import scale_one_group  # noqa: E402
from src.public_scale_probe import validate_prediction_frame  # noqa: E402


CONFIG_SHA256 = "2ef22980a0f425988bfe7acf6820a34cf19a5e2d4a1a73b048d8a9fa412f2adf"
FACTOR = np.float64(0.95)
EXPECTED_ROWS = 8760
SUPPORT_FILES = (
    "scripts/record_public_group_scale095_feedback.py",
    "tests/test_public_group_scale095_probe.py",
    "tests/test_record_public_group_scale095_feedback.py",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/public_group_scale095_probe_20260808.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/public_group_scale095_probe"),
    )
    return parser.parse_args(argv)


def _absolute(spec: Mapping[str, Any]) -> Path:
    path = Path(str(spec["path"]))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify(spec: Mapping[str, Any]) -> Path:
    path = _absolute(spec)
    if not path.is_file():
        raise FileNotFoundError(path)
    size = spec.get("bytes", spec.get("size_bytes"))
    if size is not None and path.stat().st_size != int(size):
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
    if float(config["factor"]) != float(FACTOR):
        raise AssertionError("factor differs")
    risk = config["risk_classification"]
    if risk["public_adaptive"] is not True or risk["selection_unsafe"] is not True or risk["private_champion"] is not False:
        raise AssertionError("risk disclosure differs")
    contract = config["submission_contract"]
    if contract["automatic_final_recommendation"] is not False or contract["included_in_final_top2_selection"] is not False:
        raise AssertionError("recommendation isolation differs")
    paths = {name: _verify(spec) for name, spec in config["input_identities"].items()}
    addendum = json.loads(paths["five_submission_state_addendum"].read_text(encoding="utf-8"))
    daily = addendum["daily_submission_state_after_feedback"]
    if (daily["limit"], daily["used"], daily["remaining"]) != (5, 5, 0):
        raise AssertionError("five-submission state differs")
    feedback = json.loads(paths["feedback_values"].read_text(encoding="utf-8"))
    by_label = {item["label"]: item for item in feedback["submissions"]}
    for label, frozen in (("base_v4", config["activation_gate"]["base"]), ("global_scale_095", config["activation_gate"]["global_scale_095"])):
        for component in ("score", "one_minus_nmae", "ficr"):
            if Decimal(str(by_label[label][component])) != Decimal(str(frozen[component])):
                raise AssertionError(f"feedback value differs: {label}/{component}")
    advantage = Decimal(str(by_label["global_scale_095"]["score"])) - Decimal(str(by_label["base_v4"]["score"]))
    if advantage != Decimal(config["activation_gate"]["score_advantage_decimal_exact"]) or advantage <= 0:
        raise AssertionError("activation gate differs")
    return config


def preflight(out_dir: Path) -> None:
    if out_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {out_dir}")


def _module_files(module: str) -> set[Path]:
    parts = [part for part in module.split(".") if part]
    if not parts:
        return set()
    result: set[Path] = set()
    file = PROJECT_ROOT.joinpath(*parts).with_suffix(".py")
    package = PROJECT_ROOT.joinpath(*parts, "__init__.py")
    if file.is_file():
        result.add(file.resolve())
    if package.is_file():
        result.add(package.resolve())
    for depth in range(1, len(parts)):
        initializer = PROJECT_ROOT.joinpath(*parts[:depth], "__init__.py")
        if initializer.is_file():
            result.add(initializer.resolve())
    return result


def resolve_ast_closure(entry: Path) -> tuple[Path, ...]:
    queue = [entry.resolve()]
    seen: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        if PROJECT_ROOT not in path.parents:
            raise AssertionError("closure path escapes project")
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    hits = _module_files(alias.name)
                    if hits:
                        queue.extend(hits)
                    elif (PROJECT_ROOT / alias.name.split(".")[0]).exists():
                        raise AssertionError(f"unresolved local import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative = path.relative_to(PROJECT_ROOT).with_suffix("").parts[:-1]
                    keep = len(relative) - node.level + 1
                    prefix = relative[:max(keep, 0)]
                    base = ".".join((*prefix, *((node.module or "").split("."))))
                else:
                    base = node.module or ""
                hits = _module_files(base.strip("."))
                if hits:
                    queue.extend(hits)
                else:
                    top = base.strip(".").split(".")[0]
                    if top and (PROJECT_ROOT / top).exists():
                        raise AssertionError(f"unresolved local import: {base}")
                for alias in node.names:
                    if alias.name != "*":
                        queue.extend(_module_files(f"{base}.{alias.name}".strip(".")))
    return tuple(sorted(seen))


def _snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    records = [
        {
            "name": name,
            "path": str(_absolute(spec).resolve()),
            "size_bytes": _absolute(spec).stat().st_size,
            "sha256": sha256_file(_absolute(spec)),
        }
        for name, spec in config["input_identities"].items()
    ]
    return {"schema_version": 1, "file_count": len(records), "files": records}


def _load_text_and_prediction(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    text = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    if tuple(text.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS) or len(text) != EXPECTED_ROWS:
        raise AssertionError(f"submission schema differs: {path}")
    index = pd.DatetimeIndex(pd.to_datetime(text["forecast_kst_dtm"]), name="forecast_kst_dtm")
    prediction = text.loc[:, list(TARGET_COLS)].astype(np.float64)
    prediction.index = index
    validate_prediction_frame(prediction, name=str(path))
    return text, prediction


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _probe(
    group: str,
    base_text: pd.DataFrame,
    base_prediction: pd.DataFrame,
    global_text: pd.DataFrame,
    out_dir: Path,
    stem: str,
) -> tuple[dict[str, Any], list[Path]]:
    prediction = scale_one_group(base_prediction, group, factor=float(FACTOR))
    expected_target_text = prediction[group].map(lambda value: f"{float(value):.6f}").astype("string")
    if not expected_target_text.reset_index(drop=True).equals(global_text[group].reset_index(drop=True)):
        raise AssertionError(f"target text differs from global095: {group}")
    output_text = base_text.copy(deep=True)
    output_text[group] = expected_target_text.to_numpy()
    output_text = output_text.astype("string")
    parquet_path = out_dir / f"{stem}.parquet"
    csv_path = out_dir / f"{stem}.csv"
    _atomic_parquet(prediction, parquet_path)
    _atomic_text_csv(output_text, csv_path)
    replay_prediction = pd.read_parquet(parquet_path, engine="pyarrow").astype(np.float64)
    replay_prediction.index = pd.DatetimeIndex(replay_prediction.index, name="forecast_kst_dtm")
    replay_text = pd.read_csv(csv_path, encoding="utf-8-sig", dtype="string")
    if not csv_path.read_bytes().startswith(b"\xef\xbb\xbf") or not replay_text.equals(output_text):
        raise AssertionError(f"CSV text/BOM replay differs: {group}")
    bit_identity: dict[str, bool] = {}
    text_identity: dict[str, bool] = {}
    for other in TARGET_COLS:
        if other == group:
            continue
        bit_identity[other] = bool(np.array_equal(
            replay_prediction[other].to_numpy(dtype=np.float64).view(np.uint64),
            base_prediction[other].to_numpy(dtype=np.float64).view(np.uint64),
        ))
        text_identity[other] = bool(replay_text[other].equals(base_text[other]))
    if not all(bit_identity.values()) or not all(text_identity.values()):
        raise AssertionError(f"non-target identity differs: {group}")
    if not replay_text[group].equals(global_text[group]):
        raise AssertionError(f"target/global text identity differs: {group}")
    values = replay_text.loc[:, list(TARGET_COLS)].astype(np.float64)
    if not np.isfinite(values.to_numpy()).all():
        raise AssertionError("CSV non-finite")
    return {
        "scaled_group": group,
        "factor": 0.95,
        "parquet": describe_file(parquet_path),
        "csv": describe_file(csv_path),
        "parquet_non_target_float64_bits_exact": bit_identity,
        "csv_non_target_raw_text_exact_to_base": text_identity,
        "csv_target_raw_text_exact_to_global095": True,
        "ids_timestamps_schema_rows_BOM_finite_bounds": True,
    }, [parquet_path, csv_path]


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.config = args.config.resolve()
    args.out_dir = args.out_dir.resolve()
    config = _verify_config(args.config)
    preflight(args.out_dir)
    before = _snapshot(config)
    args.out_dir.mkdir(parents=True)
    shutil.copyfile(args.config, args.out_dir / "preregister.json")
    shutil.copyfile(args.config.with_suffix(".sha256"), args.out_dir / "preregister.sha256")
    write_json_atomic(args.out_dir / "protected_inputs_before.json", before, overwrite=False)

    base_text, base_prediction = _load_text_and_prediction(_verify(config["input_identities"]["base_csv"]))
    global_text, global_prediction = _load_text_and_prediction(_verify(config["input_identities"]["global_scale095_csv"]))
    sample_text, _ = _load_text_and_prediction(_verify(config["input_identities"]["sample"]))
    identifiers = ["forecast_id", "forecast_kst_dtm"]
    if not base_text[identifiers].equals(sample_text[identifiers]) or not global_text[identifiers].equals(sample_text[identifiers]):
        raise AssertionError("identifier/timestamp identity differs")
    for group in TARGET_COLS:
        expected = np.clip(FACTOR * base_prediction[group].to_numpy(), 0.0, 1.02 * CAPACITY_KWH[group])
        observed = global_prediction[group].to_numpy()
        if np.max(np.abs(observed - expected)) > 5.1e-7:
            raise AssertionError(f"global095 formula differs: {group}")

    probes: dict[str, Any] = {}
    output_paths: list[Path] = [
        args.out_dir / "preregister.json",
        args.out_dir / "preregister.sha256",
        args.out_dir / "protected_inputs_before.json",
    ]
    for group in TARGET_COLS:
        probe, paths = _probe(
            group,
            base_text,
            base_prediction,
            global_text,
            args.out_dir,
            str(config["output_names"][group]),
        )
        probes[group] = probe
        output_paths.extend(paths)

    recorder_contract = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "recorder": config["future_feedback_recorder"]["script"],
        "required_later_feedback_count": 2,
        "accepted_groups": list(TARGET_COLS),
        "canonical_group_csvs": {
            group: probes[group]["csv"] for group in TARGET_COLS
        },
        "metric_fields": ["score", "one_minus_nmae", "ficr"],
        "inference": config["future_feedback_recorder"]["inference_decimal_exact"],
        "hybrid_rule": config["future_feedback_recorder"]["positive_group_rule"],
        "hybrid_created_now": False,
        "automatic_final_recommendation": False,
        "submission_performed": False,
    }
    recorder_path = args.out_dir / "future_feedback_recorder_contract.json"
    write_json_atomic(recorder_path, recorder_contract, overwrite=False)
    output_paths.append(recorder_path)

    results = {
        "schema_version": 1,
        "artifact_type": "public_group_scale095_probe_pack",
        "created_utc": utc_now(),
        "risk": config["risk_classification"],
        "submission_performed": False,
        "automatic_final_recommendation": False,
        "final_top2_selection_included": False,
        "activation_gate": config["activation_gate"],
        "formula": config["immutable_formula"],
        "macro_additivity": {
            **config["macro_additivity_contract"],
            "structural_column_partition_exact": True,
            "global095_reconstructed_by_taking_each_target_column_from_its_group_only_probe": True,
            "new_label_or_model_metric_values_computed": 0,
        },
        "probes": probes,
        "future_feedback": {
            **config["future_feedback_recorder"],
            "hybrid_created_now": False,
        },
    }
    results_path = args.out_dir / "results.json"
    write_json_atomic(results_path, results, overwrite=False)
    output_paths.append(results_path)

    after = _snapshot(config)
    if before != after:
        raise AssertionError("protected input changed")
    after_path = args.out_dir / "protected_inputs_after.json"
    write_json_atomic(after_path, after, overwrite=False)
    output_paths.append(after_path)

    closure = resolve_ast_closure(Path(__file__))
    support = [PROJECT_ROOT / relative for relative in SUPPORT_FILES]
    sources = [*closure, *support]
    if len(set(path.resolve() for path in sources)) != len(sources):
        raise AssertionError("duplicate source/support closure record")
    manifest = {
        "schema_version": 1,
        "artifact_type": "public_group_scale095_probe_immutable_manifest",
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA256,
        "public_adaptive": True,
        "selection_unsafe": True,
        "private_champion": False,
        "submission_performed": False,
        "automatic_final_recommendation": False,
        "final_top2_selection_included": False,
        "overwrite_guard": "new output directory required; no overwrite option",
        "activation_gate_passed": True,
        "five_submission_feedback_manifest_bound": True,
        "protected_inputs_before_after_exact": True,
        "inputs": [
            describe_file(args.config),
            describe_file(args.config.with_suffix(".sha256")),
            *[describe_file(_absolute(spec)) for spec in config["input_identities"].values()],
        ],
        "source_provenance": {
            "recursive_ast_closure": [describe_file(path) for path in closure],
            "explicit_recorder_and_tests": [describe_file(path) for path in support],
        },
        "outputs": [describe_file(path) for path in output_paths],
        "output_count_excluding_manifest": len(output_paths),
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_ROOT),
    }
    manifest_path = args.out_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest, overwrite=False)
    return {
        "out_dir": str(args.out_dir),
        "probe_count": 3,
        "csv_count": 3,
        "automatic_final_recommendation": False,
        "submission_performed": False,
        "manifest_sha256": sha256_file(manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

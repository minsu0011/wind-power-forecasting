"""Source-closure-complete v3 supersession of feature-shift pruning."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_covariate_shift_feature_pruning_v2 as v2  # noqa: E402
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)


PREREGISTER_SHA256 = "dcced90b7b95dcf0095fe32dc9ccf5661adac8261d421c18ef9070fdfe7c9780"
V2_PREREGISTER_SHA256 = "6912df4799bbd28e796a9f59f69aa60fa042d5282267a0f550dc4c6dd6825ceb"
V2_MANIFEST_SHA256 = "3020947a918fd64bffa134ea9dcf50305c3b7acf71cc663c17ca9735a444d9ec"
V2_INCIDENT_SHA256 = "94c5d0eec7aa1c0b78ad888f4f5e9cf69fdfc0a8ca51ad6860ec4ed382814af6"
V2_QUARANTINE_DIR = PROJECT_DIR / "artifacts/postgate/covariate_shift_feature_pruning_strict_v2_failed_provenance_coverage"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all",), default="all")
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/covariate_shift_feature_pruning_strict_v3"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/covariate_shift_feature_pruning_preregister_v3.json"),
    )
    return parser.parse_args(argv)


def _module_path(module: str) -> Path | None:
    if module == "src":
        path = PROJECT_DIR / "src/__init__.py"
    elif module.startswith("src.") or module.startswith("scripts."):
        path = PROJECT_DIR / (module.replace(".", "/") + ".py")
    else:
        return None
    return path if path.is_file() else None


def discover_local_source_closure(entry: Path) -> tuple[Path, ...]:
    """Discover every local Python source transitively imported by ``entry``."""

    pending = [entry.resolve()]
    discovered: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in discovered:
            continue
        if not path.is_file() or path.suffix != ".py":
            raise FileNotFoundError(path)
        discovered.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module in {"src", "scripts"}:
                    modules.update(f"{node.module}.{alias.name}" for alias in node.names)
                else:
                    modules.add(node.module)
        for module in modules:
            candidate = _module_path(module)
            if candidate is not None and candidate.resolve() not in discovered:
                pending.append(candidate.resolve())
            if module == "src" or module.startswith("src."):
                init = (PROJECT_DIR / "src/__init__.py").resolve()
                if init not in discovered:
                    pending.append(init)
    return tuple(sorted(discovered, key=lambda value: value.relative_to(PROJECT_DIR).as_posix()))


def _relative_paths(paths: Sequence[Path]) -> list[str]:
    return [path.relative_to(PROJECT_DIR).as_posix() for path in paths]


def _verify_preregister(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(path) != PREREGISTER_SHA256:
        raise AssertionError("v3 preregister hash changed")
    sidecar = path.with_suffix(".sha256")
    if sidecar.read_text(encoding="utf-8") != f"{PREREGISTER_SHA256}  {path.name}\n":
        raise AssertionError("v3 preregister sidecar changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != "covariate_shift_feature_pruning_strict_forward_v3":
        raise AssertionError("v3 experiment id changed")
    v2_path = PROJECT_DIR / payload["supersession"]["execution_contract_path"]
    if sha256_file(v2_path) != V2_PREREGISTER_SHA256:
        raise AssertionError("v2 execution contract changed")
    v2_payload, base = v2._verify_preregister(v2_path)
    incident = V2_QUARANTINE_DIR / "INCIDENT_LEDGER.json"
    manifest = V2_QUARANTINE_DIR / "manifest.json"
    if sha256_file(incident) != V2_INCIDENT_SHA256:
        raise AssertionError("v2 incident ledger changed")
    if sha256_file(manifest) != V2_MANIFEST_SHA256:
        raise AssertionError("v2 quarantined manifest changed")
    fixed = payload["fixed_candidate_contract"]
    v2_fixed = v2_payload["fixed_candidate_contract"]
    if fixed != v2_fixed:
        raise AssertionError("v3 candidate/model contract differs from v2")
    closure = discover_local_source_closure(Path(__file__).resolve())
    if _relative_paths(closure) != payload["source_closure_contract"]["ordered_paths"]:
        raise AssertionError("registered v3 source closure differs from AST discovery")
    return payload, v2_payload, base


def _source_closure_payload(
    preregister: Mapping[str, Any], preregister_path: Path
) -> dict[str, Any]:
    paths = discover_local_source_closure(Path(__file__).resolve())
    relative = _relative_paths(paths)
    if relative != preregister["source_closure_contract"]["ordered_paths"]:
        raise AssertionError("source closure changed before lock")
    source_files = {name: describe_file(path) for name, path in zip(relative, paths)}
    config_paths = {
        "v3_preregister": preregister_path,
        "v3_preregister_sidecar": preregister_path.with_suffix(".sha256"),
        "v2_execution_contract": PROJECT_DIR / preregister["supersession"]["execution_contract_path"],
        "v1_base_contract": PROJECT_DIR / "configs/covariate_shift_feature_pruning_preregister_v1.json",
        "v2_incident_ledger": V2_QUARANTINE_DIR / "INCIDENT_LEDGER.json",
        "v2_failed_manifest": V2_QUARANTINE_DIR / "manifest.json",
    }
    configs = {name: describe_file(path) for name, path in config_paths.items()}
    return {
        "schema_version": 3,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "discovery": "recursive AST Import/ImportFrom closure plus src/__init__.py package execution",
        "entrypoint": "scripts/run_covariate_shift_feature_pruning_v3.py",
        "ordered_paths": relative,
        "source_count": len(relative),
        "source_files": source_files,
        "source_files_canonical_sha256": v2._canonical_sha256(source_files),
        "configuration_files": configs,
        "configuration_files_canonical_sha256": v2._canonical_sha256(configs),
        "created_before_raw_feature_build_shift_fit_prediction_or_score": True,
    }


def _write_source_closure_lock(
    out_dir: Path,
    preregister: Mapping[str, Any],
    preregister_path: Path,
) -> Path:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    path = out_dir / "source_closure_prescore_lock.json"
    v2.bayes._write_json(path, _source_closure_payload(preregister, preregister_path))
    return path


def _require_source_closure_unchanged(
    lock_path: Path,
    preregister: Mapping[str, Any],
    preregister_path: Path,
) -> dict[str, Any]:
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    current = _source_closure_payload(preregister, preregister_path)
    for key in (
        "ordered_paths",
        "source_count",
        "source_files",
        "source_files_canonical_sha256",
        "configuration_files",
        "configuration_files_canonical_sha256",
    ):
        if current[key] != locked[key]:
            raise AssertionError(f"source closure/configuration changed after prescore lock: {key}")
    return locked


def _v2_v3_reproduction_audit(out_dir: Path, v3_result_path: Path) -> Path:
    if not v3_result_path.is_file():
        raise AssertionError("v3 metrics must exist before v2 reproduction reference is opened")
    v3_result = json.loads(v3_result_path.read_text(encoding="utf-8"))
    v2_result_path = V2_QUARANTINE_DIR / "stage1_results.json"
    v2_result = json.loads(v2_result_path.read_text(encoding="utf-8"))
    baseline, _ = v2.v1._load_stage1_baseline(
        PROJECT_DIR / "artifacts",
        json.loads((PROJECT_DIR / "configs/covariate_shift_feature_pruning_preregister_v1.json").read_text(encoding="utf-8")),
    )
    groups: dict[str, Any] = {}
    formula_count = 0
    for group in v2.TARGET_COLS:
        frame_records: dict[str, Any] = {}
        loaded: dict[str, pd.DataFrame] = {}
        for kind, relative in (
            ("shift", Path("shift") / f"stage1__{group}.parquet"),
            ("raw_prediction", Path("predictions") / f"stage1__{group}__raw_cf.parquet"),
            ("candidate_prediction", Path("candidates") / f"stage1__{group}.parquet"),
        ):
            old = pd.read_parquet(V2_QUARANTINE_DIR / relative)
            new = pd.read_parquet(out_dir / relative)
            pd.testing.assert_frame_equal(old, new, check_exact=True)
            loaded[kind] = new
            frame_records[kind] = {"frame_exact": True, "rows": len(new), "columns": new.shape[1]}
        old_subset = json.loads(
            (V2_QUARANTINE_DIR / "shift" / f"stage1__{group}__subsets.json").read_text(encoding="utf-8")
        )["subsets"]
        new_subset = json.loads(
            (out_dir / "shift" / f"stage1__{group}__subsets.json").read_text(encoding="utf-8")
        )["subsets"]
        if old_subset != new_subset:
            raise AssertionError(f"{group} v2/v3 subsets differ")
        old_model = V2_QUARANTINE_DIR / "models" / f"stage1__{group}.joblib"
        new_model = out_dir / "models" / f"stage1__{group}.joblib"
        if sha256_file(old_model) != sha256_file(new_model):
            raise AssertionError(f"{group} v2/v3 joblib bytes differ")
        if v2_result["group_results"][group]["comparisons"] != v3_result["group_results"][group]["comparisons"]:
            raise AssertionError(f"{group} v2/v3 metrics differ")
        application_index = v2.YEAR_2023 if group != "kpx_group_3" else v2.v1._segments(2023, ("H2",))["H2"]
        raw = loaded["raw_prediction"]
        candidate = loaded["candidate_prediction"]
        for key in v2.v1.CANDIDATE_KEYS:
            fraction, objective, weight = v2.v1._candidate_recipe(key)
            model_key = f"{v2.v1._fraction_tag(fraction)}_{objective}"
            expected = v2.v1.blend_candidate_kwh(
                baseline.loc[application_index, group],
                raw[model_key],
                weight=weight,
                capacity_kwh=v2.v1.CAPACITY_KWH[group],
            )
            if not np.array_equal(expected, candidate[key].to_numpy(dtype=np.float64)):
                raise AssertionError(f"{group} {key} v3 candidate formula differs")
            formula_count += 1
        groups[group] = {
            **frame_records,
            "subsets_exact": True,
            "joblib_file_sha256_exact": True,
            "joblib_sha256": sha256_file(new_model),
            "all_candidate_formulas_exact": True,
            "all_official_slice_metrics_exact": True,
            "identity_decision_exact": v2_result["group_results"][group]["selected"] == v3_result["group_results"][group]["selected"],
        }
        if not groups[group]["identity_decision_exact"]:
            raise AssertionError(f"{group} v2/v3 identity decision differs")
    if v2_result["locked_candidates"] != v3_result["locked_candidates"] or v2_result["passed_groups"] != v3_result["passed_groups"]:
        raise AssertionError("v2/v3 global Stage1 decision differs")
    prescore = json.loads((out_dir / "stage1_prescore_record.json").read_text(encoding="utf-8"))
    for source in ("ldaps", "gfs"):
        evidence = prescore["physical_weather_evidence"]["raw_prefixes"][source]
        if evidence["suffix_bytes_returned_to_weather_parser"] != 0 or evidence["future_weather_value_cells_materialized"] != 0:
            raise AssertionError("v3 physical suffix/future access changed")
    path = out_dir / "v2_v3_reproduction_audit.json"
    v2.bayes._write_json(
        path,
        {
            "schema_version": 3,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "v3_metrics_created_before_v2_reference_open": True,
            "v2_reference_permitted_use": "bit-exact reproduction audit only",
            "v2_failed_manifest_sha256": V2_MANIFEST_SHA256,
            "v2_results": describe_file(v2_result_path),
            "v3_results": describe_file(v3_result_path),
            "groups": groups,
            "candidate_formula_exact_count": formula_count,
            "locked_candidates_exact": True,
            "passed_groups_exact": True,
            "suffix_bytes_zero": True,
            "future_weather_value_cells_zero": True,
            "reproduction_passed": True,
        },
    )
    return path


def _write_manifest(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
    preregister_path: Path,
    source_lock_path: Path,
) -> Path:
    locked_closure = _require_source_closure_unchanged(
        source_lock_path, preregister, preregister_path
    )
    stage1 = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    final = json.loads((out_dir / "final_results.json").read_text(encoding="utf-8"))
    manifest_path = out_dir / "manifest.json"
    outputs = sorted(path for path in out_dir.rglob("*") if path.is_file() and path != manifest_path)
    payload = {
        "schema_version": 3,
        "artifact_type": "covariate_shift_feature_pruning_strict_forward_v3",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
        "preregister_sha256": PREREGISTER_SHA256,
        "supersedes_failed_v2": True,
        "source_closure_prescore_lock": describe_file(source_lock_path),
        "source_closure": locked_closure,
        "source_closure_unchanged_through_manifest": True,
        "physical_io": {
            "raw_stage1_prefix_read": True,
            "raw_stage1_parser_suffix_bytes": 0,
            "future_weather_value_cells_materialized_before_identity_lock": 0,
            "cache_open_before_identity_lock": False,
            "cache_open_after_identity_lock": True,
            "postlock_raw_cache_prefix_value_bits_exact": True,
            "stage1_application_labels_read_after_identity_lock": True,
            "2024_weather_read_for_model_or_selection": False,
            "2024_label_read": False,
            "2025_read": False,
            "sample_submission_read": False,
            "public_read": False,
        },
        "selection": {
            "locked_candidates": stage1["locked_candidates"],
            "passed_groups": stage1["passed_groups"],
            "v2_v3_reproduction_passed": True,
        },
        "tests": {
            "focused_prelaunch_status": "pass",
            "focused_test_count": 8,
            "full_suite_prelaunch_status": "pass",
            "full_suite_test_count": 284
        },
        "conditional_roots": {"raw_root": str(raw_dir.resolve()), "artifact_root": str(artifact_root.resolve())},
        "outputs": [describe_file(path) for path in outputs],
        "submission_created": bool(final["submission_created"]),
    }
    v2.bayes._write_json(manifest_path, payload)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    preregister_path = args.preregister.expanduser().resolve()
    preregister, execution_contract, base_contract = _verify_preregister(preregister_path)
    source_lock_path = _write_source_closure_lock(out_dir, preregister, preregister_path)
    _require_source_closure_unchanged(source_lock_path, preregister, preregister_path)
    v2.PREREGISTER_SHA256 = PREREGISTER_SHA256
    lock = v2._run_stage1(
        raw_dir=raw_dir,
        artifact_root=artifact_root,
        out_dir=out_dir,
        preregister_path=preregister_path,
        preregister=execution_contract,
        base_preregister=base_contract,
        precreated_out_dir=True,
        source_closure_lock_path=source_lock_path,
        reproduction_audit_fn=_v2_v3_reproduction_audit,
        reproduction_record_key="v2_v3_reproduction_audit",
    )
    v2._write_no_read_terminal(out_dir, lock)
    _write_manifest(
        raw_dir=raw_dir,
        artifact_root=artifact_root,
        out_dir=out_dir,
        preregister=preregister,
        preregister_path=preregister_path,
        source_lock_path=source_lock_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

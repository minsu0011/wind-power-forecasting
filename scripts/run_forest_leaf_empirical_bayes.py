"""Run the frozen posthoc ExtraTrees leaf-empirical residual Bayes candidate."""

from __future__ import annotations

import argparse
import atexit
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_full_weather_residual_pmf as parent  # noqa: E402
from src.forest_leaf_empirical_bayes import (  # noqa: E402
    FOREST_PARAMETERS,
    TRANSFER_WEIGHT,
    ExtraTreesLeafEmpiricalBayes,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.residual_histogram_bayes import ACTION_DELTAS_CF  # noqa: E402


PREREGISTER_SHA256 = "7385449b4322c3d3dc11d380e419d96ea51fd9a91dfd350f58e1b1a82f4d3be2"
V2_PREREGISTER_SHA256 = "a38f52a7f8609a7da48c1e5aa309422d79c5cd12f24531f046ea3bd346e6fca9"
V1_PREREGISTER_SHA256 = "1c9d9812814b58694dd948521dc4277051021f2cedb7bcd1904c5b98bb4e226e"
INCIDENT_SHA256 = "72e25842112b3fa3cb5945074238ad4cf4b5d9dbebef77324554be9cb0f6392e"
FAILURE_SHA256 = "94dd1a94da295cf85a8c2c0ccab7bb46552e3ef14ed8fc332f90032c147111cc"
V2_GUARD_FAILURE_SHA256 = "15edbab2dd1086a3dd5f8089e588103fb1ac86e73ac6394807964fc91765cd4e"
ARTIFACT_TYPE = "forest_leaf_empirical_residual_bayes_v3"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/forest_leaf_empirical_residual_bayes_v3"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path(
            "configs/forest_leaf_empirical_residual_bayes_preregister_v3.json"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_DIR / path


def _verify_fixed(path: Path, sha256: str, name: str) -> None:
    path = _project_path(path)
    if not path.is_file() or sha256_file(path) != sha256:
        raise AssertionError(f"{name} changed: {path}")


def _verify_preregister(path: Path) -> dict[str, Any]:
    path = _project_path(path)
    incident = PROJECT_DIR / (
        "artifacts/incidents/"
        "forest_leaf_empirical_residual_bayes_score_read_incident_v1.json"
    )
    failure = PROJECT_DIR / (
        "artifacts/incidents/"
        "forest_leaf_empirical_residual_bayes_v1_prefit_validation_failure.json"
    )
    guard_failure = PROJECT_DIR / (
        "artifacts/incidents/"
        "forest_leaf_empirical_residual_bayes_v2_guard_identity_failure.json"
    )
    v2 = PROJECT_DIR / "configs/forest_leaf_empirical_residual_bayes_preregister_v2.json"
    v1 = PROJECT_DIR / "configs/forest_leaf_empirical_residual_bayes_preregister_v1.json"
    _verify_fixed(path, PREREGISTER_SHA256, "leaf-empirical preregister")
    _verify_fixed(v1, V1_PREREGISTER_SHA256, "leaf-empirical v1 preregister")
    _verify_fixed(incident, INCIDENT_SHA256, "score-read incident")
    _verify_fixed(failure, FAILURE_SHA256, "v1 prefit failure")
    _verify_fixed(guard_failure, V2_GUARD_FAILURE_SHA256, "v2 guard failure")
    _verify_fixed(v2, V2_PREREGISTER_SHA256, "leaf-empirical v2 preregister")
    payload = json.loads(path.read_text(encoding="utf-8"))
    inherited = json.loads(v1.read_text(encoding="utf-8"))
    if payload["experiment_id"] != ARTIFACT_TYPE:
        raise AssertionError("experiment id changed")
    if inherited["forest_contract"]["parameters"] != FOREST_PARAMETERS:
        raise AssertionError("forest parameters differ from preregistration")
    action = inherited["official_utility_action"]
    if action["action_count"] != len(ACTION_DELTAS_CF):
        raise AssertionError("action grid count changed")
    if float(action["transfer_weight"]) != TRANSFER_WEIGHT:
        raise AssertionError("transfer changed")
    if inherited["feature_contract"]["total_feature_count"] != 615:
        raise AssertionError("feature count changed")
    if inherited["continuous_leaf_distribution_contract"]["kind"] != (
        "standard same-sample quantile-regression-forest weights"
    ):
        raise AssertionError("leaf weight contract changed")
    if payload["claims"]["strict_private"] is not False:
        raise AssertionError("strict-private claim must remain false")
    if payload["claims"]["selection_safe"] is not False:
        raise AssertionError("selection-safe claim must remain false")
    for spec in inherited["stage1_component_inputs"].values():
        _verify_fixed(Path(spec[0]), str(spec[2]), f"stage1 input {spec[0]}")
        if _project_path(Path(spec[0])).stat().st_size != int(spec[1]):
            raise AssertionError(f"stage1 input size changed: {spec[0]}")
    return payload


def _source_snapshot(preregister: Path) -> dict[str, Any]:
    closure = parent.protocol._static_repo_import_closure((Path(__file__).resolve(),))
    paths = {
        f"repo_source::{path.relative_to(PROJECT_DIR).as_posix()}": path
        for path in closure
    }
    preregister = _project_path(preregister).resolve()
    incident = PROJECT_DIR / (
        "artifacts/incidents/"
        "forest_leaf_empirical_residual_bayes_score_read_incident_v1.json"
    )
    failure = PROJECT_DIR / (
        "artifacts/incidents/"
        "forest_leaf_empirical_residual_bayes_v1_prefit_validation_failure.json"
    )
    guard_failure = PROJECT_DIR / (
        "artifacts/incidents/"
        "forest_leaf_empirical_residual_bayes_v2_guard_identity_failure.json"
    )
    v2 = PROJECT_DIR / "configs/forest_leaf_empirical_residual_bayes_preregister_v2.json"
    v1 = PROJECT_DIR / "configs/forest_leaf_empirical_residual_bayes_preregister_v1.json"
    paths.update(
        {
            "focused_core_test": PROJECT_DIR
            / "tests/test_forest_leaf_empirical_bayes.py",
            "focused_runner_test": PROJECT_DIR
            / "tests/test_forest_leaf_empirical_bayes_runner.py",
            "preregister": preregister,
            "preregister_sidecar": preregister.with_suffix(".sha256"),
            "incident": incident,
            "incident_sidecar": incident.with_suffix(".sha256"),
            "v1_failure": failure,
            "v1_failure_sidecar": failure.with_suffix(".sha256"),
            "v2_guard_failure": guard_failure,
            "v2_guard_failure_sidecar": guard_failure.with_suffix(".sha256"),
            "v2_preregister": v2,
            "v2_preregister_sidecar": v2.with_suffix(".sha256"),
            "v1_preregister": v1,
            "v1_preregister_sidecar": v1.with_suffix(".sha256"),
            "causal_source_preregister": PROJECT_DIR
            / "configs/catboost_residual_multiquantile_bayes_preregister_v3.json",
            "state_source_preregister": PROJECT_DIR
            / "configs/full_weather_residual_pmf_preregister_v1.json",
            "utility_source_preregister": PROJECT_DIR
            / "configs/residual_histogram_bayes_preregister.json",
        }
    )
    return {
        name: parent.shared._snapshot_file(path)
        for name, path in sorted(paths.items())
    }


def _fit_predict_locked(
    *,
    group: str,
    fit_features: pd.DataFrame,
    fit_actual: pd.Series,
    fit_baseline: pd.Series,
    application_features: pd.DataFrame,
    application_baseline: pd.Series,
    output_dir: Path,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict[str, Any], list[Path]]:
    if len(fit_features.index.intersection(application_features.index)):
        raise AssertionError("fit/application overlap")
    if not fit_features.index.max() < application_features.index.min():
        raise AssertionError("fit is not strictly before application")
    print(
        f"fit leaf empirical {group}: {fit_features.index.min()}.."
        f"{fit_features.index.max()} -> {application_features.index.min()}.."
        f"{application_features.index.max()}",
        flush=True,
    )
    model = ExtraTreesLeafEmpiricalBayes().fit(
        fit_features,
        fit_actual,
        fit_baseline,
        capacity_kwh=parent.CAPACITY_KWH[group],
    )
    action, leaf_ids, diagnostics = model.predict_action(
        application_features, application_baseline
    )
    model_path = output_dir / f"model_{group}.joblib"
    leaf_path = output_dir / f"application_leaf_ids_{group}.parquet"
    action_path = output_dir / f"raw_action_delta_cf_{group}.parquet"
    diagnostics_path = output_dir / f"action_diagnostics_{group}.parquet"
    parent.strict._atomic_joblib(model, model_path)
    parent.strict._atomic_parquet(leaf_ids, leaf_path)
    parent.strict._atomic_parquet(action.to_frame(), action_path)
    parent.strict._atomic_parquet(diagnostics, diagnostics_path)
    reloaded: ExtraTreesLeafEmpiricalBayes = joblib.load(model_path)
    action2, leaf_ids2, diagnostics2 = reloaded.predict_action(
        application_features, application_baseline
    )
    reload_checks = {
        "action_bit_exact": np.array_equal(action.to_numpy(), action2.to_numpy()),
        "leaf_ids_bit_exact": np.array_equal(
            leaf_ids.to_numpy(), leaf_ids2.to_numpy()
        ),
        "diagnostics_bit_exact": np.array_equal(
            diagnostics.to_numpy(), diagnostics2.to_numpy()
        ),
    }
    if not all(reload_checks.values()):
        raise AssertionError(f"reload changed leaf outputs for {group}: {reload_checks}")
    metadata = model.metadata()
    metadata.update(
        {
            "group": group,
            "application_start": application_features.index.min(),
            "application_end": application_features.index.max(),
            "fit_application_overlap_count": 0,
            "model_reload_checks": reload_checks,
            "qrf_weight_sum_max_abs_error": float(
                np.max(np.abs(diagnostics["qrf_weight_sum"].to_numpy() - 1.0))
            ),
        }
    )
    return action, leaf_ids, diagnostics, metadata, [
        model_path,
        leaf_path,
        action_path,
        diagnostics_path,
    ]


def _write_manifest(
    args: argparse.Namespace,
    source_before: Mapping[str, Any],
    input_snapshot: Mapping[str, Any],
    promoted: bool,
) -> Path:
    files = sorted(
        (
            path
            for path in args.out_dir.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        ),
        key=lambda path: path.relative_to(args.out_dir).as_posix(),
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "preregister": describe_file(_project_path(args.preregister)),
        "preregister_sha256": PREREGISTER_SHA256,
        "incident": describe_file(
            PROJECT_DIR
            / "artifacts/incidents/forest_leaf_empirical_residual_bayes_score_read_incident_v1.json"
        ),
        "v1_prefit_failure": describe_file(
            PROJECT_DIR
            / "artifacts/incidents/forest_leaf_empirical_residual_bayes_v1_prefit_validation_failure.json"
        ),
        "status": (
            "posthoc_dual_pass_final_created"
            if promoted
            else "posthoc_dual_reject_no_2025_read_no_csv"
        ),
        "selection_safe_claim": False,
        "strict_private_claim": False,
        "source_and_config_snapshot": source_before,
        "input_snapshot": input_snapshot,
        "outputs": [describe_file(path) for path in files],
        "output_count": len(files),
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
        "audit_contracts": {
            "candidate_count": 1,
            "model": "ExtraTreesRegressor partitions",
            "forest_parameters": FOREST_PARAMETERS,
            "distribution": "continuous_standard_same_sample_qrf",
            "action_count": len(ACTION_DELTAS_CF),
            "transfer_weight": TRANSFER_WEIGHT,
            "state_feature_count": 3,
            "candidate_before_2024_label_lock": True,
            "primary_action_delta_derived_once": True,
            "recent_uses_exact_primary_delta": True,
            "dual_all_three_group_and_mixed_all_7_strict_gate": True,
            "dual_group_and_mixed_full_components_nonnegative_gate": True,
            "no_retuning_support_rescue_group_subset_or_second_run": True,
            "final_created_only_if_complete_dual_pass": True,
        },
    }
    path = args.out_dir / "manifest.json"
    parent.strict._write_json(path, manifest)
    return path


def _configure_parent() -> None:
    parent.PREREGISTER_SHA256 = PREREGISTER_SHA256
    parent.ARTIFACT_TYPE = ARTIFACT_TYPE
    parent._verify_preregister = _verify_preregister
    parent._source_snapshot = _source_snapshot
    parent._fit_predict_locked = _fit_predict_locked
    parent._write_manifest = _write_manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    _configure_parent()
    return parent.run(args)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_parent()
    _verify_preregister(args.preregister)
    if args.preflight_only:
        occupied = parent._guard_state()
        if occupied is not None:
            raise RuntimeError(f"heavy guard occupied: {occupied}")
        output_state = "absent" if not args.out_dir.exists() else "existing_postrun"
        v1_output = PROJECT_DIR / "artifacts/postgate/forest_leaf_empirical_residual_bayes_v1"
        v1_files = len(list(v1_output.rglob("*"))) if v1_output.exists() else -1
        if v1_files != 0:
            raise AssertionError("v1 failed namespace inventory changed")
        v2_quarantine = PROJECT_DIR / (
            "artifacts/postgate/"
            "forest_leaf_empirical_residual_bayes_v2_protocol_invalid_guard_identity"
        )
        v2_files = len(list(v2_quarantine.rglob("*"))) if v2_quarantine.exists() else -1
        if v2_files != 9:
            raise AssertionError("v2 quarantine inventory changed")
        print(
            "leaf-empirical preflight PASS fit=0 predict=0 score=0 2025=0 "
            f"v1_files={v1_files} v2_quarantine_files=8 "
            f"guard_identity={ARTIFACT_TYPE} output={output_state} "
            f"prereg={PREREGISTER_SHA256} "
            f"incident={INCIDENT_SHA256}",
            flush=True,
        )
        return
    owner = parent.guardmod.acquire_heavy_guard_for_experiment(
        parent.guardmod.HEAVY_GUARD_PATH, experiment_id=ARTIFACT_TYPE
    )
    atexit.register(
        parent.guardmod.release_heavy_guard, parent.guardmod.HEAVY_GUARD_PATH, owner
    )
    print(
        f"heavy guard acquired pid={owner['pid']} prereg={PREREGISTER_SHA256}",
        flush=True,
    )
    try:
        run(args)
    finally:
        parent.guardmod.release_heavy_guard(parent.guardmod.HEAVY_GUARD_PATH, owner)


if __name__ == "__main__":
    main()

"""V8/V9 gated finalizer wrapper; frozen recipe with availability-safe paths."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "scripts/build_kma_d1_1100_wsd_final_submission.py"
V8_ROOT = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_v8"
WSD_ROOT = V8_ROOT / "wsd_typ01"
EXECUTION_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_finalizer_execution_v8.json"
MATERIALIZER_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_materializer_execution_v8.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bound(path: str) -> tuple[str, int, str]:
    p = PROJECT / path
    return path, p.stat().st_size, digest(p)


def load_frozen():
    spec = importlib.util.spec_from_file_location("_kma_finalizer_v8_base", SOURCE)
    if spec is None:
        raise RuntimeError("cannot create V8 finalizer module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    text = SOURCE.read_text(encoding="utf-8")
    old_check = "if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(a > 75) or np.any(b < 0) or np.any(b > 75):"
    new_check = "if np.isfinite(a).mean() < 0.98 or np.isfinite(b).mean() < 0.98 or np.any(a[np.isfinite(a)] < 0) or np.any(a[np.isfinite(a)] > 75) or np.any(b[np.isfinite(b)] < 0) or np.any(b[np.isfinite(b)] > 75):"
    if text.count(old_check) != 1 or text.count("atol=1e-12):") != 1:
        raise RuntimeError("frozen finalizer authorized availability sites drifted")
    text = text.replace(old_check, new_check).replace("atol=1e-12):", "atol=1e-12, equal_nan=True):")
    old_argv = '"scripts/build_kma_d1_1100_wsd_final_submission.py"'
    if text.count(old_argv) != 1:
        raise RuntimeError("frozen finalizer argv binding site drifted")
    text = text.replace(old_argv, '"scripts/build_kma_d1_1100_wsd_final_submission_v8.py"')
    exec(compile(text, str(SOURCE) + "::V8_AVAILABILITY", "exec"), module.__dict__)
    return module


base = load_frozen()
import scripts.run_kma_d1_1100_wsd_model_gate_v8 as gate_wrapper

base.__file__ = str(Path(__file__).resolve())
base.gate_runner = gate_wrapper.base
base.RECIPES = gate_wrapper.base.RECIPES
base.WSD_ROOT = WSD_ROOT
base.PREGATE_KMA = WSD_ROOT / "materialized/KMA_D1_1100_WSD_HOURLY_2022_2024_V8.parquet"
base.PREGATE_MANIFEST = WSD_ROOT / "materialized/MANIFEST_V8.json"
base.KMA_2025 = WSD_ROOT / "materialized/KMA_D1_1100_WSD_HOURLY_2025_V8.parquet"
base.KMA_2025_MANIFEST = WSD_ROOT / "materialized/MANIFEST_2025_V8.json"
base.GATE_PASS = WSD_ROOT / "MODEL_GATE_PASS_V8.json"
base.GATE_RESULTS = V8_ROOT / "model_gate/MODEL_GATE_RESULTS.json"
base.FINAL_ROOT = V8_ROOT
base.FINAL_CSV = V8_ROOT / "04_OFFICIAL_KMA_D1_WSD_AVAILABILITY_TOLERANT_V8.csv"
base.FINAL_MANIFEST = V8_ROOT / "04_OFFICIAL_KMA_D1_WSD_AVAILABILITY_TOLERANT_V8.manifest.json"
base.FINAL_MODEL_ROOT = WSD_ROOT / "final_model_v8"
base.FINAL_EXECUTION_CONFIG = EXECUTION_CONFIG
base.PREREGS = (*base.PREREGS,
    bound("configs/kma_d1_1100_wsd_availability_tolerant_v8_preregister.json"),
    bound("configs/kma_d1_1100_wsd_availability_tolerant_v9_threshold_errata.json"),
    bound("configs/kma_d1_1100_wsd_availability_tolerant_v10_timestamp_errata.json"),
)
base.BOUND["gate_runner"] = bound("scripts/run_kma_d1_1100_wsd_model_gate_v8.py")
base.BOUND["gate_execution"] = bound("configs/kma_d1_1100_wsd_model_gate_execution_v8.json")
base.BOUND["materializer"] = bound("scripts/materialize_kma_d1_1100_wsd_anchors_v8.py")


def verify_identity(record, *, under: Path | None = None):
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise RuntimeError("exact artifact identity object required")
    path = Path(str(record["path"]))
    path = path.resolve() if path.is_absolute() else (PROJECT / path).resolve()
    if under is not None and path != under.resolve() and under.resolve() not in path.parents:
        raise RuntimeError("gate artifact identity escapes model-gate root")
    observed = base.absolute_identity(path)
    if observed != {"path": path.as_posix(), "bytes": record["bytes"], "sha256": record["sha256"]}:
        raise RuntimeError(f"gate artifact identity drift: {path.name}")
    return observed


def verify_identity_tree(value) -> None:
    if isinstance(value, dict):
        if set(value) == {"path", "bytes", "sha256"}:
            verify_identity(value)
        else:
            for child in value.values():
                verify_identity_tree(child)
    elif isinstance(value, list):
        for child in value:
            verify_identity_tree(child)


_frozen_verify_execution_config = base.verify_execution_config
def verify_execution_config():
    record = _frozen_verify_execution_config()
    config = json.loads(EXECUTION_CONFIG.read_text(encoding="ascii"))
    if config.get("derived_base_finalizer") != base.relative_identity(SOURCE):
        raise RuntimeError("V8 finalizer config does not bind frozen base finalizer")
    return record
base.verify_execution_config = verify_execution_config


_frozen_validate_gate = base.validate_gate_before_any_2025_read
def validate_gate_before_any_2025_read():
    # This remains ahead of every 2025 path open and every final refit.
    gate_wrapper.verify_execution_config()
    selected, sources = _frozen_validate_gate()
    out = V8_ROOT / "model_gate"
    input_path = out / "INPUT_LOCK_BEFORE_MODEL_FIT.json"
    input_lock = json.loads(input_path.read_text(encoding="ascii"))
    if set(input_lock) != {"bound", "public_scores_used", "kma_2025_read"}:
        raise RuntimeError("model input-lock schema drift")
    if input_lock.get("public_scores_used") is not False or input_lock.get("kma_2025_read") is not False:
        raise RuntimeError("model input-lock blind provenance failed")
    expected_input_names = {
        *{f"preregister_v{i}" for i in range(1, len(base.gate_runner.PREREGS) + 1)},
        *base.gate_runner.IDENTITIES,
        *{f"weather_{group}" for group in base.gate_runner.WEATHER},
        "materialized",
    }
    if set(input_lock.get("bound", {})) != expected_input_names:
        raise RuntimeError("model input-lock bound-input census drift")
    verify_identity_tree(input_lock.get("bound"))
    results = json.loads(base.GATE_RESULTS.read_text(encoding="ascii"))
    freeze_path = out / "PREDICTION_FREEZE_BEFORE_METRICS.json"
    if results.get("prediction_lock") != base.absolute_identity(freeze_path):
        raise RuntimeError("gate result does not bind exact global prediction lock")
    lock = json.loads(freeze_path.read_text(encoding="ascii"))
    if lock.get("status") != "ALL_RECIPE_FOLD_GROUP_PREDICTIONS_AND_BASELINES_FROZEN_BEFORE_ANY_METRIC_COMPUTATION":
        raise RuntimeError("global prediction-lock status drift")
    if any(lock.get(k) is not False for k in ("public_scores_read_or_used", "kma_2025_read", "csv_created")):
        raise RuntimeError("global prediction-lock provenance failed")
    label_access = lock.get("label_access_before_lock", {})
    if label_access != {
        "training_year_prefixes_only_per_fold": True,
        "fold_1_training_operating_years": [2022],
        "fold_2_training_operating_years": [2022, 2023],
        "application_year_metrics_computed_before_lock": False,
        "application_labels_may_be_reused_as_later_fold_training_only_after_their_fold_freeze": True,
        "full_2022_2024_label_frame_opened_before_global_lock": False,
    }:
        raise RuntimeError("global prediction-lock label-access ledger drift")
    expected_folds = set(base.gate_runner.FOLDS)
    expected_predictions = {f"{fold}/{recipe}" for fold in expected_folds for recipe in base.RECIPES}
    if set(lock.get("predictions", {})) != expected_predictions or set(lock.get("models", {})) != expected_predictions:
        raise RuntimeError("global candidate/model artifact census drift")
    if set(lock.get("baseline_predictions", {})) != expected_folds or set(lock.get("fold_freezes", {})) != expected_folds:
        raise RuntimeError("global baseline/fold-lock census drift")
    for section in ("predictions", "models", "baseline_predictions", "fold_freezes"):
        for identity_record in lock[section].values():
            verify_identity(identity_record, under=out)
    for fold in expected_folds:
        fold_path = out / "fold_locks" / f"{fold}__FREEZE.json"
        if lock["fold_freezes"][fold] != base.absolute_identity(fold_path):
            raise RuntimeError("global lock/fold-lock identity mismatch")
        fold_lock = json.loads(fold_path.read_text(encoding="ascii"))
        if fold_lock.get("fold") != fold or fold_lock.get("metrics_computed") is not False:
            raise RuntimeError("fold-lock semantics drift")
        fold_spec = base.gate_runner.FOLDS[fold]
        if fold_lock.get("training_operating_years_decoded") != list(fold_spec["train_years"]) or fold_lock.get("application_operating_year") != fold_spec["apply_year"]:
            raise RuntimeError("fold-lock train/application-year ledger drift")
        if fold_lock.get("baseline") != lock["baseline_predictions"][fold]:
            raise RuntimeError("fold-lock baseline binding drift")
        if fold_lock.get("predictions") != {r: lock["predictions"][f"{fold}/{r}"] for r in base.RECIPES}:
            raise RuntimeError("fold-lock prediction binding drift")
        if fold_lock.get("models") != {r: lock["models"][f"{fold}/{r}"] for r in base.RECIPES}:
            raise RuntimeError("fold-lock model binding drift")
    sources["model_input_lock"] = base.absolute_identity(input_path)
    sources["global_prediction_lock"] = base.absolute_identity(freeze_path)
    return selected, sources
base.validate_gate_before_any_2025_read = validate_gate_before_any_2025_read


_frozen_load_kma_2025 = base.load_kma_2025_after_gate
def load_kma_2025_after_gate():
    frame, identities = _frozen_load_kma_2025()
    manifest = json.loads(base.KMA_2025_MANIFEST.read_text(encoding="ascii"))
    if manifest.get("materializer") != base.relative_identity(Path(base.BOUND["materializer"][0])):
        raise RuntimeError("2025 manifest does not bind current V8 materializer wrapper")
    evidence = manifest.get("source_evidence", {})
    if base.relative_identity(MATERIALIZER_CONFIG) not in evidence.get("chain_identities", []):
        raise RuntimeError("2025 manifest does not bind current V8 materializer config")
    coverage = evidence.get("coverage", {}).get("2025", {})
    if coverage.get("coverage_gate_minimum") != 0.98 or coverage.get("coverage_gate_pass") is not True:
        raise RuntimeError("2025 manifest 0.98 coverage gate absent or failed")
    clock = frame.index - base.pd.Timedelta(hours=1)
    leads = clock.hour.to_numpy() + 1
    days = clock.normalize()
    for name in ("wsd_cell_94_121", "wsd_cell_94_122"):
        values = frame[name].to_numpy(dtype=float)
        manifest_annual = coverage[f"cell_{name.removeprefix('wsd_cell_')}_annual_finite_fraction"]
        actual_annual = float(np.isfinite(values).mean())
        if actual_annual < 0.98 or manifest_annual != actual_annual:
            raise RuntimeError(f"2025 annual cell coverage mismatch: {name}")
        manifest_leads = coverage[f"cell_{name.removeprefix('wsd_cell_')}_per_lead_finite_fraction"]
        for lead in range(1, 25):
            actual = float(np.isfinite(values[leads == lead]).mean())
            if actual < 0.98 or manifest_leads.get(str(lead)) != actual:
                raise RuntimeError(f"2025 per-lead cell coverage mismatch: {name}/{lead}")
        for day in base.pd.unique(days):
            finite_count = int(np.isfinite(values[days == day]).sum())
            if finite_count not in (0, 24):
                raise RuntimeError(f"2025 partial operating-day cell availability: {name}/{day}")
    return frame, identities
base.load_kma_2025_after_gate = load_kma_2025_after_gate


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()

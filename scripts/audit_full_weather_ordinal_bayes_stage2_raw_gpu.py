"""Independent read-only audit of the rejected ordinal-Bayes G3 Stage2.

The audit replays the stored model on 2024 weather features, recomputes the
registered action and both official-metric gates, and independently resolves the
local Python import closure.  It never reads 2025/sample values or creates a CSV.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_ficr_bayes_decision_strict as strict  # noqa: E402
from scripts import run_full_weather_ordinal_bayes_stage2_final as producer  # noqa: E402
from src.full_weather_ordinal_bayes import BIN_COUNT, FullWeatherOrdinalBayes  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


CANONICAL = ROOT / "artifacts/postgate/full_weather_ordinal_bayes_strict_v1"
REPORT = ROOT / "artifacts/audits/full_weather_ordinal_bayes_g3_stage2_v1_raw_gpu.json"
STAGE1_MANIFEST = CANONICAL / "manifest.json"
STAGE2_MANIFEST = CANONICAL / "manifest_stage2_final_g3.json"
PREREGISTER = ROOT / "configs/full_weather_ordinal_bayes_preregister_v1.json"
RUNNER = ROOT / "scripts/run_full_weather_ordinal_bayes_stage2_final.py"
REVERSE_PATCH = (
    ROOT
    / "artifacts/incidents/full_weather_ordinal_bayes_stage2_postrun_verifier/"
    "current_b3e817_to_executed_8ab4.reverse.patch"
)
POSTRUN_AUDIT = ROOT / "artifacts/audits/full_weather_ordinal_bayes_g3_stage2_postrun_v2.json"

EXPECTED = {
    "stage1_manifest": "29786f6826f9a4873bec1d2781653aba2edad2a5b037e8ed023b38f88e594efe",
    "stage2_manifest": "a443c2ab714053d0a2f23b6f7de5d21e1498ae6161c61341df93835b7c2325e1",
    "preregister": "0052188e1ee6ed1a61cc458e620f357451d2ec6bdb9faf304156f1e185c82f44",
    "executed_runner": "8ab4bfb985b2e51b77461e63dafd8c1df2f0032c48b0457a67e96f2678daac06",
    "executed_runner_bytes": 31_322,
    "current_runner": "b3e81735dd0bcaf04290579247e2d9431446b69ca1276394b05cabd26e466416",
    "reverse_patch": "d4c2efc4dfc069bb1841942b8fdeeeb45021b84bcdc85f40546c24e920d522a3",
    "postrun_audit": "30d1a6ff389149493b570628d5705968dc1c8686822f395d9d0c0281d1d61ae0",
    "src_init": "4b449e4e899a7505c6f38c74cd951b3365b03f194c15b12c9a0a8b9b0cb30739",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def describe(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_record(record: Mapping[str, Any]) -> None:
    path = Path(record["path"])
    if not path.is_file():
        raise AssertionError(f"recorded file missing: {path}")
    if path.stat().st_size != int(record["size_bytes"]) or sha256(path) != record["sha256"]:
        raise AssertionError(f"recorded file changed: {path}")


def canonical_snapshot() -> dict[str, dict[str, Any]]:
    return {
        rel(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(CANONICAL.rglob("*"), key=lambda item: str(item).lower())
        if path.is_file()
    }


def module_files(module: str) -> set[Path]:
    if not module:
        return set()
    parts = module.split(".")
    found: set[Path] = set()
    file_candidate = ROOT.joinpath(*parts).with_suffix(".py")
    package_candidate = ROOT.joinpath(*parts, "__init__.py")
    if file_candidate.is_file():
        found.add(file_candidate.resolve())
    if package_candidate.is_file():
        found.add(package_candidate.resolve())
    for depth in range(1, len(parts)):
        initializer = ROOT.joinpath(*parts[:depth], "__init__.py")
        if initializer.is_file():
            found.add(initializer.resolve())
    return found


def independent_ast_closure(entrypoint: Path) -> set[Path]:
    queue = [entrypoint.resolve()]
    visited: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        if ROOT.resolve() not in path.parents:
            raise AssertionError(f"local closure escaped root: {path}")
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    queue.extend(module_files(alias.name) - visited)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative = path.relative_to(ROOT).with_suffix("").parts[:-1]
                    keep = len(relative) - node.level + 1
                    prefix = relative[: max(keep, 0)]
                    base = ".".join((*prefix, *((node.module or "").split("."))))
                else:
                    base = node.module or ""
                queue.extend(module_files(base.strip(".")) - visited)
                for alias in node.names:
                    if alias.name != "*":
                        queue.extend(module_files(f"{base}.{alias.name}".strip(".")) - visited)
    return visited


def apply_unified_patch_in_memory(source: bytes, patch: bytes) -> bytes:
    source_lines = source.decode("utf-8").splitlines(keepends=True)
    patch_lines = patch.decode("utf-8").splitlines(keepends=True)
    hunk = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    offset = 0
    cursor = 0
    applied = 0
    while cursor < len(patch_lines):
        match = hunk.match(patch_lines[cursor])
        if not match:
            cursor += 1
            continue
        old_start = int(match.group(1))
        old_count = int(match.group(2) or "1")
        cursor += 1
        old_chunk: list[str] = []
        new_chunk: list[str] = []
        while cursor < len(patch_lines) and not patch_lines[cursor].startswith("@@ "):
            line = patch_lines[cursor]
            if line.startswith(("--- ", "+++ ")):
                break
            if line.startswith("\\"):
                cursor += 1
                continue
            marker = line[:1]
            payload = line[1:]
            if marker in {" ", "-"}:
                old_chunk.append(payload)
            if marker in {" ", "+"}:
                new_chunk.append(payload)
            cursor += 1
        if len(old_chunk) != old_count:
            raise AssertionError("reverse-patch old hunk count changed")
        position = old_start - 1 + offset
        if source_lines[position : position + len(old_chunk)] != old_chunk:
            raise AssertionError("reverse-patch preimage differs from current runner")
        source_lines[position : position + len(old_chunk)] = new_chunk
        offset += len(new_chunk) - len(old_chunk)
        applied += 1
    if applied != 1:
        raise AssertionError(f"reverse-patch hunk count changed: {applied}")
    return "".join(source_lines).encode("utf-8")


def arrays_equal(left: pd.DataFrame | pd.Series, right: pd.DataFrame | pd.Series) -> bool:
    return left.index.equals(right.index) and np.array_equal(left.to_numpy(), right.to_numpy())


def metric_summary(comparisons: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for baseline_name in ("primary_v3", "recent_v4"):
        record = comparisons[baseline_name]
        output[baseline_name] = {
            "g3_total_score_deltas": {
                segment: float(record["group_g3"][segment]["delta"])
                for segment in producer.REQUIRED_SEGMENTS
            },
            "mixed_total_score_deltas": {
                segment: float(record["mixed"][segment]["delta"])
                for segment in producer.REQUIRED_SEGMENTS
            },
            "g3_full_delta_one_minus_nmae": float(record["gate"]["group_full_delta_one_minus_nmae"]),
            "g3_full_delta_ficr": float(record["gate"]["group_full_delta_ficr"]),
            "mixed_full_delta_one_minus_nmae": float(record["gate"]["mixed_full_delta_one_minus_nmae"]),
            "mixed_full_delta_ficr": float(record["gate"]["mixed_full_delta_ficr"]),
            "g3_all_7_positive": bool(
                record["gate"]["group_all_7_total_score_deltas_strictly_positive"]
            ),
            "mixed_all_7_positive": bool(
                record["gate"]["mixed_all_7_total_score_deltas_strictly_positive"]
            ),
            "passed": bool(record["gate"]["passed"]),
        }
    return output


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    if sha256(STAGE1_MANIFEST) != EXPECTED["stage1_manifest"]:
        raise AssertionError("Stage1 manifest changed")
    if sha256(STAGE2_MANIFEST) != EXPECTED["stage2_manifest"]:
        raise AssertionError("Stage2 append manifest changed")
    if sha256(PREREGISTER) != EXPECTED["preregister"]:
        raise AssertionError("preregister changed")
    before = canonical_snapshot()

    stage1_manifest = json.loads(STAGE1_MANIFEST.read_text(encoding="utf-8"))
    stage2_manifest = json.loads(STAGE2_MANIFEST.read_text(encoding="utf-8"))
    if stage2_manifest["status"] != "stage2_reject_no_final_csv":
        raise AssertionError("Stage2 status changed")

    for record in stage1_manifest["source_and_config_snapshot"].values():
        verify_record(record)
    for record in stage1_manifest["census_snapshot"].values():
        verify_record(record)
    for record in stage1_manifest["outputs"]:
        verify_record(record)

    historical_runner_record = stage2_manifest["source_and_config_snapshot"][
        "repo_source::scripts/run_full_weather_ordinal_bayes_stage2_final.py"
    ]
    for key, record in stage2_manifest["source_and_config_snapshot"].items():
        if key != "repo_source::scripts/run_full_weather_ordinal_bayes_stage2_final.py":
            verify_record(record)
    for records in stage2_manifest["input_snapshot"].values():
        for record in records.values():
            verify_record(record)
    for record in stage2_manifest["outputs"]:
        verify_record(record)
    if stage2_manifest["input_snapshot"]["final_after_stage2_promotion"] != {}:
        raise AssertionError("rejected run snapshotted final inputs")

    if sha256(RUNNER) != EXPECTED["current_runner"] or sha256(REVERSE_PATCH) != EXPECTED["reverse_patch"]:
        raise AssertionError("postrun source lineage changed")
    reconstructed = apply_unified_patch_in_memory(RUNNER.read_bytes(), REVERSE_PATCH.read_bytes())
    reconstructed_hash = hashlib.sha256(reconstructed).hexdigest()
    if (
        len(reconstructed) != EXPECTED["executed_runner_bytes"]
        or reconstructed_hash != EXPECTED["executed_runner"]
        or historical_runner_record["size_bytes"] != len(reconstructed)
        or historical_runner_record["sha256"] != reconstructed_hash
    ):
        raise AssertionError("executed runner reconstruction differs")
    if sha256(POSTRUN_AUDIT) != EXPECTED["postrun_audit"]:
        raise AssertionError("postrun supersession audit changed")

    ast_closure = independent_ast_closure(RUNNER)
    manifest_closure = {
        Path(record["path"]).resolve()
        for key, record in stage2_manifest["source_and_config_snapshot"].items()
        if key.startswith("repo_source::")
    }
    ast_unbound = sorted(rel(path) for path in ast_closure - manifest_closure)
    manifest_extra = sorted(rel(path) for path in manifest_closure - ast_closure)
    if ast_unbound != ["src/__init__.py"] or manifest_extra:
        raise AssertionError(
            f"unexpected independent AST closure difference: unbound={ast_unbound}, extra={manifest_extra}"
        )
    src_init = ROOT / "src/__init__.py"
    if src_init.stat().st_size != 69 or sha256(src_init) != EXPECTED["src_init"]:
        raise AssertionError("unbound package initializer changed")
    if ast.parse(src_init.read_text(encoding="utf-8")).body[0].__class__.__name__ != "Expr":
        raise AssertionError("src initializer is no longer docstring-only")

    expected_canonical = {
        Path(record["path"]).resolve() for record in stage1_manifest["outputs"]
    } | {STAGE1_MANIFEST.resolve()}
    expected_canonical |= {
        Path(record["path"]).resolve() for record in stage2_manifest["outputs"]
    } | {STAGE2_MANIFEST.resolve()}
    observed_canonical = {path.resolve() for path in CANONICAL.rglob("*") if path.is_file()}
    canonical_extra = sorted(rel(path) for path in observed_canonical - expected_canonical)
    canonical_missing = sorted(rel(path) for path in expected_canonical - observed_canonical)
    if canonical_extra or canonical_missing:
        raise AssertionError(f"canonical output closure differs extra={canonical_extra} missing={canonical_missing}")

    stage2_dir = CANONICAL / "stage2"
    primary_base = pd.read_parquet(stage2_dir / "primary_baseline_2024.parquet")
    recent_base = pd.read_parquet(stage2_dir / "recent_baseline_2024.parquet")
    primary_candidate = pd.read_parquet(stage2_dir / "primary_candidate_2024.parquet")
    recent_candidate = pd.read_parquet(stage2_dir / "recent_same_delta_candidate_2024.parquet")
    stored_delta = pd.read_parquet(stage2_dir / "primary_delta_cf_2024.parquet").iloc[:, 0]
    stored_action = pd.read_parquet(stage2_dir / "raw_action_cf_g3.parquet").iloc[:, 0]
    stored_probability = pd.read_parquet(stage2_dir / "probability42_g3.parquet")
    stored_diagnostics = pd.read_parquet(stage2_dir / "action_diagnostics_g3.parquet")
    if stored_probability.shape != (8_784, BIN_COUNT):
        raise AssertionError("42-bin probability shape changed")

    expected_primary, expected_recent, expected_delta = producer._same_delta_candidate(
        primary_base, recent_base, stored_action
    )
    formula_checks = {
        "primary_candidate_bit_exact": arrays_equal(primary_candidate, expected_primary),
        "recent_same_primary_delta_candidate_bit_exact": arrays_equal(recent_candidate, expected_recent),
        "primary_delta_bit_exact": arrays_equal(stored_delta, expected_delta),
    }
    identities: dict[str, bool] = {}
    for baseline_name, baseline, candidate in (
        ("primary_v3", primary_base, primary_candidate),
        ("recent_v4", recent_base, recent_candidate),
    ):
        for group in TARGET_COLS[:2]:
            identities[f"{baseline_name}/{group}"] = (
                baseline[group].to_numpy().tobytes() == candidate[group].to_numpy().tobytes()
            )
    if not all(formula_checks.values()) or not all(identities.values()):
        raise AssertionError("stored formula or G1/G2 identity differs")

    independent_primary_g3 = np.clip(
        0.90 * primary_base[producer.G3].to_numpy(dtype=np.float64) / CAPACITY_KWH[producer.G3]
        + 0.10 * stored_action.to_numpy(dtype=np.float64),
        0.0,
        1.02,
    ) * CAPACITY_KWH[producer.G3]
    independent_delta = (
        independent_primary_g3 - primary_base[producer.G3].to_numpy(dtype=np.float64)
    ) / CAPACITY_KWH[producer.G3]
    independent_recent_g3 = np.clip(
        recent_base[producer.G3].to_numpy(dtype=np.float64) / CAPACITY_KWH[producer.G3]
        + independent_delta,
        0.0,
        1.02,
    ) * CAPACITY_KWH[producer.G3]

    model: FullWeatherOrdinalBayes = joblib.load(stage2_dir / "model_g3.joblib")
    application_index = pd.DatetimeIndex(primary_base.index, name="forecast_kst_dtm")
    expected_index = strict._year_index(2024)
    if not application_index.equals(expected_index):
        raise AssertionError("stored application index changed")
    feature_cache = ROOT / "artifacts/cache/kpx_group_3_weather_train.parquet"
    application_features = producer._read_feature_cache(feature_cache, application_index)
    replay_action, replay_probability, replay_diagnostics = model.predict_action(
        application_features, primary_base[producer.G3]
    )
    model_replay = {
        "action_bit_exact": arrays_equal(replay_action, stored_action),
        "probability42_bit_exact": arrays_equal(replay_probability, stored_probability),
        "diagnostics_bit_exact": arrays_equal(replay_diagnostics, stored_diagnostics),
        "probability_shape": list(stored_probability.shape),
        "probability_min": float(stored_probability.to_numpy().min()),
        "probability_row_sum_max_abs_error": float(
            np.max(np.abs(stored_probability.sum(axis=1).to_numpy() - 1.0))
        ),
        "model_metadata": model.metadata(),
    }
    if not all(model_replay[key] for key in ("action_bit_exact", "probability42_bit_exact", "diagnostics_bit_exact")):
        raise AssertionError("joblib replay differs from locked arrays")

    prescore_lock_path = CANONICAL / "stage2_2024_dual_prescore_lock.json"
    prescore_lock = json.loads(prescore_lock_path.read_text(encoding="utf-8"))
    for record in prescore_lock["candidate_outputs"]:
        verify_record(record)
    if not (
        prescore_lock["created_before_any_2024_application_label_value"]
        and prescore_lock["model_probability_action_primary_delta_and_dual_candidates_frozen"]
        and prescore_lock["g1_g2_bit_identity"]
        and prescore_lock["recent_uses_exact_primary_delta"]
    ):
        raise AssertionError("candidate-before-label lock contract changed")

    result_path = CANONICAL / "stage2_results_g3.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    labels = strict._read_full_labels(Path(r"data/local/open/train/train_labels.csv"))
    dual_passed, recomputed = producer._dual_gate(
        labels,
        {"primary_v3": primary_base, "recent_v4": recent_base},
        {"primary_v3": primary_candidate, "recent_v4": recent_candidate},
        2024,
    )
    metric_hash = strict._canonical_sha256(recomputed)
    if metric_hash != result["comparisons_sha256"] or recomputed != result["dual_baseline_comparisons"]:
        raise AssertionError("official metric replay differs")
    promotion_lock = json.loads((CANONICAL / "stage2_promotion_lock_g3.json").read_text(encoding="utf-8"))
    if dual_passed or result["candidate_promoted"] or promotion_lock["candidate_promoted"]:
        raise AssertionError("rejected candidate promotion changed")

    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_full_weather_ordinal_bayes.py",
            "tests/test_full_weather_ordinal_bayes_runner.py",
            "tests/test_full_weather_ordinal_bayes_stage2_final.py",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if tests.returncode != 0 or "23 passed" not in tests.stdout:
        raise AssertionError(f"focused tests failed: {tests.stdout}\n{tests.stderr}")

    if (CANONICAL / "final").exists() or list(CANONICAL.rglob("*.csv")):
        raise AssertionError("rejected candidate has final/CSV")
    after = canonical_snapshot()
    if before != after:
        raise AssertionError("canonical changed during read-only audit")

    summary = metric_summary(recomputed)
    report = {
        "schema_version": 1,
        "audit_id": "full_weather_ordinal_bayes_g3_stage2_v1_raw_gpu",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": {
            "numeric_model_formula_metric_integrity": "PASS",
            "performance": "REJECT_STAGE2_RECENT_V4",
            "strict_source_closure": "FAIL_PROVENANCE_ONLY_UNBOUND_DOCSTRING_PACKAGE_INITIALIZER",
            "promotion": False,
            "final_2025_or_csv": "NOT_CREATED",
            "overall": "PASS_NUMERIC_REPRODUCTION_REJECT_PERFORMANCE_WITH_PROVENANCE_CAVEAT",
        },
        "identity": {
            "preregister": describe(PREREGISTER),
            "stage1_manifest": describe(STAGE1_MANIFEST),
            "stage2_append_manifest": describe(STAGE2_MANIFEST),
            "stage2_prescore_lock": describe(prescore_lock_path),
            "stage2_results": describe(result_path),
            "stage2_promotion_lock": describe(CANONICAL / "stage2_promotion_lock_g3.json"),
        },
        "source_and_output_closure": {
            "independent_AST_file_count": len(ast_closure),
            "manifest_repo_source_file_count": len(manifest_closure),
            "AST_relative_paths": sorted(rel(path) for path in ast_closure),
            "manifest_unregistered_executable_files": ast_unbound,
            "manifest_extra_files": manifest_extra,
            "unbound_initializer": {
                **describe(src_init),
                "content_classification": "module_docstring_only_no_runtime_import_or_assignment",
                "numeric_candidate_impact": "none detected; separate provenance-only defect",
            },
            "executed_runner_reconstruction": {
                "historical_manifest_record": historical_runner_record,
                "current_runner": describe(RUNNER),
                "reverse_patch": describe(REVERSE_PATCH),
                "reconstructed_bytes": len(reconstructed),
                "reconstructed_sha256": reconstructed_hash,
                "exact": True,
                "postrun_supersession_audit": describe(POSTRUN_AUDIT),
            },
            "stage1_source_config_census_outputs_current": True,
            "stage2_sources_except_documented_runner_current": True,
            "stage2_inputs_and_outputs_current": True,
            "canonical_union_extra_files": canonical_extra,
            "canonical_union_missing_files": canonical_missing,
            "canonical_file_count": len(observed_canonical),
        },
        "candidate_before_label_lock": {
            "lock_hash_exact": True,
            "candidate_output_count": len(prescore_lock["candidate_outputs"]),
            "model_probability_action_primary_delta_and_dual_candidates_frozen": True,
            "created_before_any_2024_application_label_value": True,
            "G1_G2_identity_locked": True,
            "same_primary_delta_locked": True,
        },
        "model_and_probability_replay": model_replay,
        "formula_replay": {
            **formula_checks,
            "G1_G2_float_bits_identity": identities,
            "independent_primary_formula_max_abs_difference_kwh": float(
                np.max(np.abs(independent_primary_g3 - primary_candidate[producer.G3].to_numpy()))
            ),
            "independent_delta_max_abs_difference_cf": float(
                np.max(np.abs(independent_delta - stored_delta.to_numpy()))
            ),
            "independent_recent_formula_max_abs_difference_kwh": float(
                np.max(np.abs(independent_recent_g3 - recent_candidate[producer.G3].to_numpy()))
            ),
            "formula": {
                "primary": "clip(.90*primary_baseline_cf + .10*raw_action_cf, 0, 1.02)",
                "recent": "clip(recent_baseline_cf + exact_primary_delta_cf, 0, 1.02)",
            },
        },
        "official_2024_metric_replay": {
            "comparison_sha256": metric_hash,
            "stored_comparison_exact": True,
            "registered_comparison_records": 28,
            "segments": list(producer.REQUIRED_SEGMENTS),
            "summary": summary,
            "primary_v3_passed": summary["primary_v3"]["passed"],
            "recent_v4_passed": summary["recent_v4"]["passed"],
            "dual_gate_passed": dual_passed,
            "recent_failed_slices": [
                segment
                for segment, delta in summary["recent_v4"]["g3_total_score_deltas"].items()
                if delta <= 0.0
            ],
        },
        "future_and_nonmutation": {
            "audit_2025_or_sample_read": False,
            "fit_calls": 0,
            "candidate_files_created": 0,
            "CSV_files_created": 0,
            "canonical_modified": False,
            "canonical_before_equals_after": before == after,
            "final_directory_exists": False,
            "canonical_CSV_count": 0,
            "no_reselection_retune_or_rescue": True,
        },
        "tests": {
            "command": (
                ".venv/Scripts/python.exe -m pytest -q tests/test_full_weather_ordinal_bayes.py "
                "tests/test_full_weather_ordinal_bayes_runner.py "
                "tests/test_full_weather_ordinal_bayes_stage2_final.py"
            ),
            "passed": 23,
            "failed": 0,
            "stdout_tail": tests.stdout.strip().splitlines()[-1],
        },
        "risk": {
            "leaderboard_score_claim": False,
            "private_champion_claim": False,
            "selection_unsafe": False,
            "candidate_rejected": True,
            "source_closure_defect_requires_no_candidate_action_because_no_2025_or_CSV_exists": True,
        },
        "audit_implementation": describe(Path(__file__).resolve()),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": rel(REPORT), "bytes": REPORT.stat().st_size, "sha256": sha256(REPORT)}))


if __name__ == "__main__":
    main()

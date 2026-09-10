from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_public_adaptive_scale097_g2_delta as runner
from src.manifest import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]
V1 = PROJECT_DIR / "configs/public_adaptive_scale097_g2_delta_preregister_v1.json"
V2 = PROJECT_DIR / "configs/public_adaptive_scale097_g2_delta_preregister_v2.json"


def test_v2_hash_supersession_and_risk() -> None:
    assert sha256_file(V1) == runner.V1_SHA
    assert sha256_file(V2) == runner.CONFIG_SHA
    payload = json.loads(V2.read_text(encoding="utf-8"))
    assert payload["supersession_reason"]["v1_A_transformed_value_cells_materialized"] == 0
    assert payload["supersession_reason"]["v1_B_transformed_value_cells_materialized"] == 0
    assert payload["supersession_reason"]["v1_executable"] is False
    assert payload["risk_classification"]["public_adaptive"] is True
    assert payload["risk_classification"]["selection_unsafe"] is True
    assert payload["risk_classification"]["private_champion"] is False


def test_factor_is_one_rounded_probe_without_vertex() -> None:
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    assert float(runner.FACTOR) == 0.97
    assert v2["immutable_candidates_repeated"]["factor_count"] == 1
    assert v2["immutable_candidates_repeated"]["factor"] == 0.97
    assert v2["immutable_candidates_repeated"]["exact_quadratic_vertex_or_fitted_optimum_forbidden"] is True
    assert v1["public_feedback_used"]["no_response_surface_grid"] is True


def test_recursive_ast_closure_is_exact_and_minimal() -> None:
    closure = runner.resolve_ast_closure(
        PROJECT_DIR / "scripts/run_public_adaptive_scale097_g2_delta.py"
    )
    assert [path.relative_to(PROJECT_DIR).as_posix() for path in closure] == [
        "scripts/run_public_adaptive_scale097_g2_delta.py",
        "src/__init__.py",
        "src/manifest.py",
    ]


def test_candidate_formula_execution_order_and_identity() -> None:
    index = pd.date_range("2025-01-01 01:00", periods=3, freq="h")
    base = pd.DataFrame(
        {
            "kpx_group_1": [100.0, 200.0, 300.0],
            "kpx_group_2": [1000.0, 2000.0, 3000.0],
            "kpx_group_3": [400.0, 500.0, 600.0],
        }, index=index, dtype=np.float64,
    )
    rescue = base.copy()
    rescue.iloc[1, 1] = np.float64(0.98) * rescue.iloc[1, 1]
    A, B, delta = runner.candidates(base, rescue)
    for group in runner.TARGETS:
        expected_A = np.clip(np.float64(0.97) * base[group].to_numpy(), 0, 1.02 * runner.CAPACITY[group])
        expected_B = np.clip(
            expected_A + (rescue[group].to_numpy() - base[group].to_numpy()),
            0, 1.02 * runner.CAPACITY[group],
        )
        assert np.array_equal(A[group].to_numpy().view(np.uint64), expected_A.view(np.uint64))
        assert np.array_equal(B[group].to_numpy().view(np.uint64), expected_B.view(np.uint64))
    assert np.flatnonzero(delta[runner.GROUP].to_numpy() != 0).tolist() == [1]
    for group in ("kpx_group_1", "kpx_group_3"):
        assert np.array_equal(A[group].to_numpy().view(np.uint64), B[group].to_numpy().view(np.uint64))


def test_diagnostic_is_comparison_only_and_cannot_retune() -> None:
    payload = json.loads(V2.read_text(encoding="utf-8"))
    diagnostic = payload["frozen_2024_interaction_diagnostic_repeated"]
    assert diagnostic["only_comparison"] == "B2024 minus A2024"
    assert diagnostic["result_cannot_change_artifacts"] is True
    assert payload["immutable_candidates_repeated"]["no_new_gate_factor_threshold_delta_blend_grid_retry_or_tuning"] is True
    assert payload["immutable_candidates_repeated"]["both_CSVs_materialized_regardless_of_diagnostic"] is True


def test_default_output_is_v2_namespace() -> None:
    args = runner.parse_args([])
    assert args.out_dir.as_posix().endswith("public_adaptive_scale097_g2_delta_v2")


def test_existing_output_directory_is_rejected_before_materialization(tmp_path: Path) -> None:
    existing = tmp_path / "already_exists"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already_exists"):
        runner.main(["--out-dir", str(existing)])

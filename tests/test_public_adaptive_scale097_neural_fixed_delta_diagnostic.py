from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_public_adaptive_scale097_neural_fixed_delta_diagnostic as runner
from src.manifest import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/public_adaptive_scale097_neural_fixed_delta_diagnostic_preregister_v1.json"


def test_config_hash_risk_and_immutable_rule() -> None:
    assert sha256_file(CONFIG) == runner.CONFIG_SHA
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["risk_classification"]["public_adaptive"] is True
    assert payload["risk_classification"]["selection_unsafe"] is True
    assert payload["risk_classification"]["private_champion"] is False
    rule = payload["immutable_single_interaction"]
    assert rule["scale_inherited_and_unchanged"] == 0.97
    assert rule["neural_weight_inherited_and_unchanged"] == 0.05
    assert rule["neural_margin_inherited_and_unchanged"] == 0.01
    assert rule["gates_inherited_and_unchanged"] == runner.EXPECTED_DELTA_ROWS
    assert rule["no_delta_scale_gate_weight_margin_surface_lookup_threshold_formula_grid_blend_retry_or_rescue_change"] is True


def test_AST_closure_is_exact_and_minimal() -> None:
    closure = runner.resolve_ast_closure(ROOT / "scripts/run_public_adaptive_scale097_neural_fixed_delta_diagnostic.py")
    assert [path.relative_to(ROOT).as_posix() for path in closure] == [
        "scripts/run_public_adaptive_scale097_neural_fixed_delta_diagnostic.py",
        "src/__init__.py",
        "src/manifest.py",
    ]


def test_compose_fixed_delta_and_g3_bit_identity() -> None:
    index = pd.date_range("2024-01-01 01:00", periods=2, freq="h")
    A = pd.DataFrame([[100.0, 200.0, 300.0], [22031.0, 400.0, 500.0]], index=index, columns=runner.TARGETS)
    delta = pd.DataFrame([[10.0, -20.0, 0.0], [100.0, 30.0, 0.0]], index=index, columns=runner.TARGETS)
    candidate = runner.compose(A, delta)
    assert candidate.iloc[0, 0] == 110.0
    assert candidate.iloc[0, 1] == 180.0
    assert candidate.iloc[1, 0] == 1.02 * runner.CAPACITY[runner.TARGETS[0]]
    assert np.array_equal(candidate[runner.IDENTITY].to_numpy().view(np.uint64), A[runner.IDENTITY].to_numpy().view(np.uint64))


def test_diagnostic_requires_all_groups_mixed_and_components() -> None:
    index = runner._year_index()
    actual = pd.DataFrame({group: np.full(len(index), 0.5 * runner.CAPACITY[group]) for group in runner.TARGETS}, index=index)
    A = pd.DataFrame({group: np.full(len(index), 0.4 * runner.CAPACITY[group]) for group in runner.TARGETS}, index=index)
    candidate = actual.copy()
    candidate[runner.IDENTITY] = A[runner.IDENTITY]
    assert runner.diagnostic(actual, A, candidate)["diagnostic_GO"] is True
    candidate[runner.TARGETS[1]] = A[runner.TARGETS[1]]
    assert runner.diagnostic(actual, A, candidate)["diagnostic_GO"] is False


def test_NO_GO_contract_forbids_final_and_CSV() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert "no final refit" in payload["frozen_2024_gate"]["NO_GO"]
    assert payload["execution_integrity"]["output_CSV"] is False
    assert payload["post_GO_only"]["CSV_only_after_separate_final_prescore"] is True


def test_default_output_is_postrun_safe() -> None:
    args = runner.parse_args([])
    assert args.out_dir.as_posix().endswith("public_adaptive_scale097_neural_fixed_delta_diagnostic_v1")


def test_existing_output_rejected_before_values(tmp_path: Path) -> None:
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="exists"):
        runner.main(["--out-dir", str(existing)])
    assert list(existing.iterdir()) == []

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_public_adaptive_scale097_ficr_g1_delta as runner
from src.manifest import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_DIR / "configs/public_adaptive_scale097_ficr_g1_delta_preregister_v1.json"


def test_config_hash_risk_and_single_candidate_are_frozen() -> None:
    assert sha256_file(CONFIG) == runner.CONFIG_SHA
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    risk = payload["risk_classification"]
    candidate = payload["immutable_single_candidate"]
    assert risk["public_adaptive"] is True
    assert risk["selection_unsafe"] is True
    assert risk["strict_final_isolation"] is False
    assert risk["private_champion"] is False
    assert candidate["candidate_count"] == 1
    assert candidate["selected_group"] == "kpx_group_1 only"
    assert candidate["smoothed_variant_forbidden"] is True
    assert candidate["no_coefficient_threshold_gate_factor_grid_blend_retry_or_retuning"] is True


def test_recursive_ast_closure_is_exact_and_minimal() -> None:
    closure = runner.resolve_ast_closure(
        PROJECT_DIR / "scripts/run_public_adaptive_scale097_ficr_g1_delta.py"
    )
    assert [path.relative_to(PROJECT_DIR).as_posix() for path in closure] == [
        "scripts/run_public_adaptive_scale097_ficr_g1_delta.py",
        "src/__init__.py",
        "src/manifest.py",
    ]


def test_compose_uses_exact_fixed_delta_and_preserves_identity_bits() -> None:
    index = pd.date_range("2025-01-01 01:00", periods=3, freq="h")
    A = pd.DataFrame(
        {
            "kpx_group_1": [100.0, 22031.0, 300.0],
            "kpx_group_2": [1000.0, 2000.0, 3000.0],
            "kpx_group_3": [400.0, 500.0, 600.0],
        },
        index=index,
        dtype=np.float64,
    )
    reference = pd.DataFrame(
        {
            "kpx_group_1": [90.0, 100.0, 400.0],
            "kpx_group_2": [1.0, 2.0, 3.0],
            "kpx_group_3": [4.0, 5.0, 6.0],
        },
        index=index,
        dtype=np.float64,
    )
    adjusted = reference.copy()
    adjusted["kpx_group_1"] = [100.0, 200.0, 350.0]
    candidate, delta = runner.compose(A, reference, adjusted)
    expected = np.clip(
        A[runner.GROUP].to_numpy() + (adjusted[runner.GROUP] - reference[runner.GROUP]).to_numpy(),
        0.0,
        1.02 * runner.CAPACITY[runner.GROUP],
    )
    assert np.array_equal(candidate[runner.GROUP].to_numpy().view(np.uint64), expected.view(np.uint64))
    assert np.array_equal(
        delta[runner.GROUP].to_numpy().view(np.uint64),
        (adjusted[runner.GROUP] - reference[runner.GROUP]).to_numpy().view(np.uint64),
    )
    for group in runner.IDENTITY:
        assert np.array_equal(
            candidate[group].to_numpy().view(np.uint64),
            A[group].to_numpy().view(np.uint64),
        )


def test_compose_rejects_non_g1_delta() -> None:
    index = pd.date_range("2025-01-01 01:00", periods=1, freq="h")
    A = pd.DataFrame([[1.0, 2.0, 3.0]], index=index, columns=runner.TARGETS)
    reference = A.copy()
    adjusted = reference.copy()
    adjusted.loc[index[0], "kpx_group_2"] += 1.0
    with pytest.raises(AssertionError, match="delta not G1-only"):
        runner.compose(A, reference, adjusted)


def test_diagnostic_contract_is_all_seven_segments_and_cannot_retune() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    diagnostic = payload["frozen_2024_diagnostic"]
    assert diagnostic["segments"] == list(runner.SEGMENTS)
    assert diagnostic["only_comparison"] == "B minus A"
    assert diagnostic["result_cannot_change_formula"] is True
    assert payload["immutable_single_candidate"]["CSV_materialized_regardless_of_diagnostic"] is True


def test_protected_snapshot_covers_scale097_and_both_ficr_trees() -> None:
    snapshot = runner.protected_upstream_snapshot()
    assert snapshot["protected_roots"] == list(runner.PROTECTED_UPSTREAM_ROOTS)
    assert snapshot["protected_files"] == list(runner.PROTECTED_UPSTREAM_FILES)
    assert snapshot["file_count"] > 0
    names = {item["relative_path"] for item in snapshot["files"]}
    assert "artifacts/postgate/public_adaptive_scale097_g2_delta_v2/manifest_frozen_postrun_v2.json" in names
    assert "artifacts/postgate/ficr_bayes_v4_composition/ficr_bayes_recent_v4_2025.parquet" in names
    assert "artifacts/postgate/ficr_bayes_decision/oof/stage2_fixed_candidate_2024.parquet" in names
    assert "artifacts/final_cf_fix/corrected_recent_v4.csv" in names


def test_default_output_namespace_is_postrun_safe() -> None:
    args = runner.parse_args([])
    assert args.out_dir.as_posix().endswith("public_adaptive_scale097_ficr_g1_delta_v1")


def test_existing_output_directory_is_rejected_before_materialization(tmp_path: Path) -> None:
    existing = tmp_path / "already_exists"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already_exists"):
        runner.main(["--out-dir", str(existing)])
    assert list(existing.iterdir()) == []

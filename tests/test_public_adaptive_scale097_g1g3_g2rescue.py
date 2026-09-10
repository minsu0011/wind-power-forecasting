from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from scripts import run_public_adaptive_scale097_g1g3_g2rescue as runner


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2024-01-01 01:00", periods=4, freq="h", name="forecast_kst_dtm")
    base = pd.DataFrame(
        {
            "kpx_group_1": [1000.0, 2000.0, 3000.0, 4000.0],
            "kpx_group_2": [5000.0, 6000.0, 7000.0, 8000.0],
            "kpx_group_3": [9000.0, 10000.0, 11000.0, 12000.0],
        },
        index=index,
        dtype=np.float64,
    )
    rescue = base.copy(deep=True)
    rescue["kpx_group_2"] = [4900.0, 6000.0, 6860.0, 8000.0]
    return base, rescue


def test_composition_uses_plain_g1_g3_and_exact_rescue_g2() -> None:
    base, rescue = _frames()
    plain, candidate = runner.compose(base, rescue)
    for group in runner.TARGETS:
        np.testing.assert_array_equal(
            plain[group].to_numpy(),
            np.float64(0.97) * base[group].to_numpy(),
        )
    for group in runner.IDENTITY_TO_PLAIN:
        np.testing.assert_array_equal(
            candidate[group].to_numpy().view(np.uint64),
            plain[group].to_numpy().view(np.uint64),
        )
    np.testing.assert_array_equal(
        candidate[runner.ACTIVE_GROUP].to_numpy().view(np.uint64),
        rescue[runner.ACTIVE_GROUP].to_numpy().view(np.uint64),
    )


def test_composition_is_not_the_prior_scale_plus_delta_formula() -> None:
    base, rescue = _frames()
    plain, candidate = runner.compose(base, rescue)
    prior = plain[runner.ACTIVE_GROUP].to_numpy() + (
        rescue[runner.ACTIVE_GROUP].to_numpy() - base[runner.ACTIVE_GROUP].to_numpy()
    )
    assert not np.array_equal(
        candidate[runner.ACTIVE_GROUP].to_numpy().view(np.uint64),
        np.ascontiguousarray(prior).view(np.uint64),
    )


def test_reference_evaluation_contract_record_counts_and_strictness() -> None:
    index = runner.parent._year_index(2024)
    actual = pd.DataFrame(10000.0, index=index, columns=runner.TARGETS)
    reference = pd.DataFrame(11500.0, index=index, columns=runner.TARGETS)
    candidate = actual.copy(deep=True)
    all_groups = runner._evaluate_reference(actual, reference, candidate, runner.TARGETS)
    assert len(all_groups["group_records"]) == 21
    assert len(all_groups["mixed_records"]) == 7
    assert all_groups["passed"] is True
    g2_only = runner._evaluate_reference(actual, reference, candidate, (runner.ACTIVE_GROUP,))
    assert len(g2_only["group_records"]) == 7
    assert len(g2_only["mixed_records"]) == 7
    assert g2_only["passed"] is True


def test_config_hash_namespace_risk_and_dual_reference_gate_are_frozen() -> None:
    args = runner.parse_args([])
    assert args.out_dir.as_posix() == "artifacts/postgate/public_adaptive_scale097_g1g3_g2rescue_v1"
    assert hashlib.sha256(Path(args.config).read_bytes()).hexdigest() == runner.CONFIG_SHA256
    config = runner._verify_config(Path(args.config))
    risk = config["risk_classification"]
    assert all(risk[name] for name in ("public_adaptive", "posthoc_2024", "multiple_testing", "selection_unsafe"))
    diagnostic = config["diagnostic_2024"]
    assert "G1, G2, and G3" in diagnostic["comparison_vs_unmodified_baseline"]["group_gate"]
    assert "G2 total-score" in diagnostic["comparison_vs_plain_scale097"]["active_group_gate"]
    assert "bit-exact" in diagnostic["comparison_vs_plain_scale097"]["identity_gate"]


def test_recursive_ast_closure_contains_runner_parent_and_manifest() -> None:
    closure = runner._source_closure(runner.parse_args([]))
    paths = set(closure["resolved_relative_paths"])
    assert "scripts/run_public_adaptive_scale097_g1g3_g2rescue.py" in paths
    assert "scripts/run_public_adaptive_scale097_g2_delta.py" in paths
    assert "src/manifest.py" in paths
    assert closure["unresolved_local_imports"] == []


def test_existing_output_directory_is_rejected_postrun_safely() -> None:
    args = runner.parse_args([])
    with tempfile.TemporaryDirectory() as temporary:
        args.out_dir = Path(temporary)
        with pytest.raises(FileExistsError, match="existing output directory"):
            runner.run(args)


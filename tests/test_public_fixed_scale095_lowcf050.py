from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_public_fixed_scale095_lowcf050 as runner


ROOT = Path(__file__).resolve().parents[1]


def test_preregister_hash_and_single_rule() -> None:
    path = ROOT / "configs/public_fixed_scale095_lowcf050_preregister_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == runner.CONFIG_SHA256
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["immutable_single_rule"]["factor"] == 0.95
    assert config["immutable_single_rule"]["cutoff_cf"] == 0.5
    assert config["immutable_single_rule"]["candidate_count"] == 1
    assert config["stage1_and_stage2"]["no_post_result_group_rescue_or_retry"]


def test_rule_scales_only_registered_low_side_and_preserves_bits() -> None:
    index = pd.date_range("2023-01-01 01:00", periods=5, freq="h")
    capacities = np.array([21_600.0, 21_600.0, 21_000.0])
    fractions = np.array([0.0, 0.25, 0.50, 0.5000000001, 0.90])
    values = fractions[:, None] * capacities[None, :]
    baseline = pd.DataFrame(values, index=index, columns=runner.TARGET_COLS)
    candidate, mask = runner._apply_rule(baseline)
    assert mask.iloc[:3].to_numpy().all()
    assert not mask.iloc[3:].to_numpy().any()
    np.testing.assert_array_equal(
        candidate.iloc[:3].to_numpy(), baseline.iloc[:3].to_numpy() * 0.95
    )
    assert np.array_equal(
        candidate.iloc[3:].to_numpy().view(np.uint64),
        baseline.iloc[3:].to_numpy().view(np.uint64),
    )


def test_parent_contract_is_hash_bound_and_inherits_reader_order() -> None:
    config = json.loads(
        (ROOT / "configs/public_fixed_scale095_lowcf050_preregister_v1.json").read_text(
            encoding="utf-8"
        )
    )
    merged = runner._merged_parent_contract(config)
    assert merged["stage1_2023"]["candidate_before_label_lock"].startswith(
        "Materialize and hash"
    )
    assert merged["stage2_2024_if_and_only_if_stage1_passes"][
        "candidate_before_label_lock"
    ].startswith("Materialize and hash")
    assert merged["final_if_and_only_if_dual_stage2_passes"]["csv"] == (
        "public_fixed_scale095_lowcf050_2025.csv"
    )


def test_duplicate_census_has_no_exact_execution() -> None:
    census = json.loads(
        (
            ROOT
            / "artifacts/audits/public_fixed_scale095_lowcf050_duplicate_census_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert census["exact_execution_found"] is False
    assert census["candidate_cells_computed_before_this_census"] == 0
    assert census["candidate_metric_values_computed_before_this_census"] == 0

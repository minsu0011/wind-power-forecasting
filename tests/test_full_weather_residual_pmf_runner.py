from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_full_weather_residual_pmf as runner
from src.metric import CAPACITY_KWH, TARGET_COLS


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_preregister_hash_and_fixed_contract_verify() -> None:
    config = runner._verify_preregister(
        PROJECT_DIR / "configs/full_weather_residual_pmf_preregister_v1.json"
    )
    assert runner.PREREGISTER_SHA256 == (
        "5c091c8dfdc0c9c8b854132eab9e2ba1162813d4157be65506fb0546e93cdbea"
    )
    assert config["official_utility_action"]["fixed_transfer_weight"] == 0.10
    assert config["classifier"]["parameters"]["random_state"] == 42


def test_default_output_namespace_is_fresh_contract_without_mutation() -> None:
    args = runner.parse_args([])
    assert args.out_dir.as_posix() == "artifacts/postgate/full_weather_residual_pmf_v1"
    assert not args.preflight_only


def test_source_orders_candidate_lock_before_full_2024_label_parse() -> None:
    source = (PROJECT_DIR / "scripts/run_full_weather_residual_pmf.py").read_text(
        encoding="utf-8"
    )
    stage2_start = source.index("def _run_stage2(")
    stage2_end = source.index("def _sample(", stage2_start)
    block = source[stage2_start:stage2_end]
    assert block.index("_write_stage2_candidates(") < block.index(
        "strict._read_full_labels("
    )
    candidate_start = source.index("def _write_stage2_candidates(")
    candidate_end = source.index("def _run_stage2(", candidate_start)
    candidate_block = source[candidate_start:candidate_end]
    assert "strict._read_full_labels(" not in candidate_block
    assert "application_2024_label_value_cells_materialized\": 0" in candidate_block


def test_recent_candidate_uses_only_exact_primary_delta_all_groups() -> None:
    index = pd.date_range(
        "2024-01-01 01:00", periods=4, freq="h", name="forecast_kst_dtm"
    )
    primary = pd.DataFrame(index=index, columns=TARGET_COLS, dtype=float)
    recent = pd.DataFrame(index=index, columns=TARGET_COLS, dtype=float)
    actions: dict[str, pd.Series] = {}
    for position, group in enumerate(TARGET_COLS):
        capacity = CAPACITY_KWH[group]
        primary[group] = capacity * (0.2 + 0.1 * position)
        recent[group] = capacity * (0.25 + 0.1 * position)
        actions[group] = pd.Series(
            np.asarray([-0.15, -0.05, 0.05, 0.15]), index=index
        )
    primary_candidate, recent_candidate, delta = (
        runner._same_primary_delta_candidates(primary, recent, actions)
    )
    for group in TARGET_COLS:
        capacity = CAPACITY_KWH[group]
        expected_delta = (
            primary_candidate[group].to_numpy() - primary[group].to_numpy()
        ) / capacity
        assert np.array_equal(delta[group].to_numpy(), expected_delta)
        expected_recent = np.clip(recent[group].to_numpy() / capacity + expected_delta, 0, 1.02) * capacity
        assert np.array_equal(recent_candidate[group].to_numpy(), expected_recent)


def test_stage2_input_snapshot_source_contains_no_public_scale_paths() -> None:
    source = (PROJECT_DIR / "scripts/run_full_weather_residual_pmf.py").read_text(
        encoding="utf-8"
    )
    start = source.index("def _stage2_input_snapshot(")
    stop = source.index("def _final_input_snapshot(", start)
    block = source[start:stop].lower()
    assert "public_scale" not in block
    assert "submission.csv" not in block
    assert "2025" not in block


def test_preflight_namespace_is_postrun_safe(tmp_path: Path) -> None:
    existing = tmp_path / "canonical"
    existing.mkdir()
    runner.main(["--preflight-only", "--out-dir", str(existing)])


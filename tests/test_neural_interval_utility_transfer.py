from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_neural_interval_utility_transfer as runner
from src.manifest import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_DIR / "configs/neural_interval_utility_transfer_preregister_v2.json"


def test_config_hash_risk_and_exact_rule() -> None:
    assert sha256_file(CONFIG) == runner.CONFIG_SHA
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    risk = payload["risk_classification"]
    rule = payload["immutable_single_rule"]
    assert risk["selection_unsafe"] is True
    assert risk["public_adaptive"] is False
    assert risk["strict_final_isolation"] is False
    assert risk["private_champion"] is False
    assert rule["blend_weight"] == 0.05
    assert rule["utility_advantage_margin"] == 0.01
    assert rule["groups_with_rule"] == list(runner.GROUPS)
    assert rule["identity_groups"] == [runner.IDENTITY]
    assert rule["no_grid_retry_or_rescue"] is True


def test_recursive_ast_closure_contains_all_core_dependencies() -> None:
    closure = runner.resolve_ast_closure(PROJECT_DIR / "scripts/run_neural_interval_utility_transfer.py")
    names = {path.relative_to(PROJECT_DIR).as_posix() for path in closure}
    assert {
        "scripts/run_neural_interval_utility_transfer.py",
        "scripts/run_raw_grid_wind_lgb.py",
        "src/direct_interval_probability.py",
        "src/manifest.py",
        "src/metric.py",
        "src/raw_spatiotemporal_attention.py",
    }.issubset(names)
    assert all(path.is_file() for path in closure)


def test_linear_utility_interpolation_exact() -> None:
    grid = np.arange(103, dtype=np.float64)
    surface = np.vstack((grid, 2.0 * grid, -grid))
    query = np.array([0.123, 0.999, 1.02])
    observed = runner.interpolated_utility(surface, query)
    assert np.allclose(observed, [12.3, 199.8, -102.0], rtol=0.0, atol=1e-13)


def test_compose_uses_primary_gate_same_delta_and_each_baseline_g3_identity() -> None:
    index = pd.date_range("2024-01-01 01:00", periods=3, freq="h", name="forecast_kst_dtm")
    primary = pd.DataFrame(
        {
            "kpx_group_1": np.array([0.50, 0.50, 0.50]) * 21600.0,
            "kpx_group_2": np.array([0.40, 0.40, 0.40]) * 21600.0,
            "kpx_group_3": [100.0, 200.0, 300.0],
        }, index=index,
    )
    interaction = primary.copy()
    interaction["kpx_group_1"] += 10.0
    interaction["kpx_group_2"] -= 20.0
    interaction["kpx_group_3"] += 30.0
    neural = pd.DataFrame(
        {"kpx_group_1": [1.0, 0.5, 0.0], "kpx_group_2": [0.4, 0.9, 0.4], "kpx_group_3": [0.1, 0.2, 0.3]},
        index=index,
    )
    action_grid = np.arange(103, dtype=np.float64) / 100.0
    utilities = {
        "kpx_group_1": np.tile(4.0 * action_grid, (3, 1)),
        "kpx_group_2": np.tile(4.0 * action_grid, (3, 1)),
    }
    candidate_p, candidate_i, delta, diagnostics = runner.compose_stage2(primary, interaction, neural, utilities)
    for group in runner.GROUPS:
        gate = diagnostics[group]["gate"].to_numpy()
        assert np.array_equal(
            (candidate_i[group] - interaction[group]).to_numpy()[gate],
            delta[group].to_numpy()[gate],
        )
        assert np.array_equal(candidate_i[group].to_numpy()[~gate], interaction[group].to_numpy()[~gate])
    assert np.array_equal(candidate_p[runner.IDENTITY].to_numpy().view(np.uint64), primary[runner.IDENTITY].to_numpy().view(np.uint64))
    assert np.array_equal(candidate_i[runner.IDENTITY].to_numpy().view(np.uint64), interaction[runner.IDENTITY].to_numpy().view(np.uint64))


def test_gate_is_strict_greater_than() -> None:
    advantages = np.array([runner.MARGIN, np.nextafter(runner.MARGIN, np.inf)])
    assert (advantages > runner.MARGIN).tolist() == [False, True]


def test_promotion_requires_all_21_deltas_and_full_components() -> None:
    index = runner._segments(2024)["full"]
    actual = pd.DataFrame(
        {group: np.full(len(index), 0.50 * runner.CAPACITY_KWH[group]) for group in runner.TARGET_COLS},
        index=index,
    )
    baseline = pd.DataFrame(
        {group: np.full(len(index), 0.40 * runner.CAPACITY_KWH[group]) for group in runner.TARGET_COLS},
        index=index,
    )
    candidate = actual.copy()
    candidate[runner.IDENTITY] = baseline[runner.IDENTITY]
    result = runner.score_candidate(actual, baseline, candidate)
    assert result["passed"] is True
    failed = runner.score_candidate(actual, baseline, baseline.copy())
    assert failed["passed"] is False


def test_reader_order_keeps_full_2024_labels_out_of_prescore() -> None:
    assert "_read_label_prefix" in runner._prescore.__code__.co_names
    assert "_read_full_labels" not in runner._prescore.__code__.co_names
    assert "_read_full_labels" in runner._score_final.__code__.co_names
    assert "candidate_before_2024_label_lock.json" in runner._score_final.__code__.co_consts


def test_default_output_namespace_is_postrun_safe() -> None:
    args = runner.parse_args(["--stage", "prescore"])
    assert args.out_dir.as_posix().endswith("neural_interval_utility_transfer_v2")


def test_existing_output_rejected_before_input_or_fit(tmp_path: Path) -> None:
    existing = tmp_path / "already_exists"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already_exists"):
        runner.main(["--stage", "prescore", "--out-dir", str(existing)])
    assert list(existing.iterdir()) == []

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_action_uplift_policy as runner
from src.action_uplift_policy import (
    ACTION_FACTORS,
    CLASSIFIER_PARAMETERS,
    MODEL_FEATURE_COLUMNS,
    PROBABILITY_THRESHOLD,
    REGRESSOR_PARAMETERS,
    ActionUpliftPolicy,
    compare_group_blocks,
    exact_row_utility_deltas,
    expanded_action_design,
    passes_all_blocks,
    transfer_delta,
)
from src.manifest import sha256_file
from src.metric import CAPACITY_KWH, group_metrics
from src.residual_histogram_bayes import META_FEATURE_COLUMNS


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _features(index: pd.DatetimeIndex, base_cf: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(991)
    output = pd.DataFrame(
        rng.normal(0.0, 0.2, (len(index), len(META_FEATURE_COLUMNS))),
        index=index,
        columns=META_FEATURE_COLUMNS,
    )
    output.loc[:, "base_cf"] = base_cf
    component_columns = [
        "lgb_l1_cf",
        "lgb_q07_cf",
        "shared_l1_cf",
        "shared_q07_cf",
        "top200_q07_cf",
        "energy_q06_cf",
    ]
    component = np.column_stack(
        [np.clip(base_cf + rng.normal(0.0, 0.03, len(index)), 0.0, 1.02) for _ in component_columns]
    )
    output.loc[:, component_columns] = component
    output.loc[:, "component_mean_cf"] = component.mean(axis=1)
    output.loc[:, "component_std_cf"] = component.std(axis=1)
    output.loc[:, "component_min_cf"] = component.min(axis=1)
    output.loc[:, "component_max_cf"] = component.max(axis=1)
    output.loc[:, "component_range_cf"] = component.max(axis=1) - component.min(axis=1)
    output.loc[:, "lgb_q07_minus_lgb_l1_cf"] = component[:, 1] - component[:, 0]
    output.loc[:, "shared_q07_minus_shared_l1_cf"] = component[:, 3] - component[:, 2]
    return output.astype(np.float64)


def test_v2_preregister_hash_and_supersession_contract() -> None:
    path = PROJECT_DIR / "configs/action_uplift_policy_preregister_v2.json"
    assert sha256_file(path) == runner.PREREGISTER_SHA256
    assert sha256_file(PROJECT_DIR / "configs/action_uplift_policy_preregister_v1.json") == runner.BASE_PREREGISTER_SHA256
    merged, evidence = runner._load_preregister(path)
    assert merged["experiment_id"] == "action_uplift_policy_strict_v2"
    assert merged["stage1_component_identities"]["top200_q07"]["sha256"] == (
        "2d907b9704987c6ff58cfd8caf6bcbad659603f527783f628f2863a471dc4426"
    )
    v2 = json.loads(path.read_text(encoding="utf-8"))
    assert v2["supersedes"]["v1_action_uplift_fit_count"] == 0
    assert v2["supersedes"]["v1_candidate_score_count"] == 0
    assert set(evidence) == {"v1", "v2", "v2_sidecar"}


def test_fixed_actions_threshold_and_model_parameters_are_exact() -> None:
    assert ACTION_FACTORS == (0.98, 0.95)
    assert PROBABILITY_THRESHOLD == 0.55
    assert CLASSIFIER_PARAMETERS["objective"] == "binary"
    assert CLASSIFIER_PARAMETERS["random_state"] == 5801
    assert REGRESSOR_PARAMETERS["objective"] == "regression_l1"
    assert REGRESSOR_PARAMETERS["random_state"] == 5802
    for parameters in (CLASSIFIER_PARAMETERS, REGRESSOR_PARAMETERS):
        assert parameters["num_leaves"] == 7
        assert parameters["max_depth"] == 3
        assert parameters["min_child_samples"] == 240
        assert parameters["reg_alpha"] == 2.0
        assert parameters["reg_lambda"] == 24.0


def test_expanded_design_is_row_major_and_exact_schema() -> None:
    index = pd.date_range("2023-01-01 01:00", periods=2, freq="h")
    features = _features(index, np.array([0.50, 0.70]))
    design = expanded_action_design(features)
    assert design.shape == (4, len(MODEL_FEATURE_COLUMNS))
    action_position = len(META_FEATURE_COLUMNS)
    np.testing.assert_array_equal(design[:, action_position], [0.98, 0.95, 0.98, 0.95])
    np.testing.assert_allclose(design[:, action_position + 2], [0.49, 0.475, 0.686, 0.665])
    np.testing.assert_array_equal(design[0, : len(META_FEATURE_COLUMNS)], design[1, : len(META_FEATURE_COLUMNS)])


def test_row_targets_match_nmae_ficr_and_total_components_separately() -> None:
    rng = np.random.default_rng(82)
    index = pd.date_range("2023-01-01 01:00", periods=1200, freq="h")
    capacity = CAPACITY_KWH["kpx_group_1"]
    actual_cf = rng.uniform(0.10, 0.95, len(index))
    baseline_cf = np.clip(actual_cf + rng.normal(0.015, 0.075, len(index)), 0.0, 1.02)
    actual = pd.Series(actual_cf * capacity, index=index)
    baseline = pd.Series(baseline_cf * capacity, index=index)
    eligible, total_delta, metadata = exact_row_utility_deltas(
        baseline, actual, capacity_kwh=capacity
    )
    assert eligible.all()
    mean_actual = actual_cf.mean()
    base_error = np.abs(baseline_cf - actual_cf)
    base_payment = np.select([base_error <= 0.06, base_error <= 0.08], [4.0, 3.0], default=0.0)
    base_metrics = group_metrics(actual, baseline, capacity)
    for position, factor in enumerate(ACTION_FACTORS):
        action_cf = np.clip(factor * baseline_cf, 0.0, 1.02)
        action = pd.Series(action_cf * capacity, index=index)
        action_error = np.abs(action_cf - actual_cf)
        action_payment = np.select(
            [action_error <= 0.06, action_error <= 0.08], [4.0, 3.0], default=0.0
        )
        row_nmae = base_error - action_error
        row_ficr = (actual_cf / mean_actual) * (action_payment - base_payment) / 4.0
        action_metrics = group_metrics(actual, action, capacity)
        exact_nmae = action_metrics.one_minus_nmae - base_metrics.one_minus_nmae
        exact_ficr = action_metrics.ficr - base_metrics.ficr
        assert np.isclose(row_nmae.mean(), exact_nmae, rtol=0.0, atol=5e-15)
        assert np.isclose(row_ficr.mean(), exact_ficr, rtol=0.0, atol=5e-15)
        assert np.isclose(
            total_delta[:, position].mean(),
            0.5 * exact_nmae + 0.5 * exact_ficr,
            rtol=0.0,
            atol=5e-15,
        )
    assert metadata["maximum_exactness_error"] <= 5e-15


def test_classifier_and_regression_targets_are_finite_and_balanced() -> None:
    index = pd.date_range("2023-01-01 01:00", periods=1000, freq="h")
    capacity = CAPACITY_KWH["kpx_group_2"]
    base_cf = np.linspace(0.30, 0.85, len(index))
    offset = np.where(np.arange(len(index)) % 2 == 0, -0.055, 0.055)
    actual_cf = np.clip(base_cf + offset, 0.10, 1.00)
    baseline = pd.Series(base_cf * capacity, index=index)
    actual = pd.Series(actual_cf * capacity, index=index)
    eligible, target, _ = exact_row_utility_deltas(baseline, actual, capacity_kwh=capacity)
    flattened = target.reshape(-1)
    binary = flattened > 0.0
    assert eligible.sum() == len(index)
    assert np.isfinite(flattened).all()
    assert binary.sum() >= 25
    assert (~binary).sum() >= 25


class _FakeClassifier:
    def __init__(self, positive: np.ndarray) -> None:
        self.positive = np.asarray(positive, dtype=np.float64)

    def predict_proba(self, _: np.ndarray) -> np.ndarray:
        return np.column_stack((1.0 - self.positive, self.positive))


class _FakeRegressor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float64)

    def predict(self, _: np.ndarray) -> np.ndarray:
        return self.values


def test_policy_selection_threshold_tie_break_and_identity_bit_exact() -> None:
    fit_index = pd.date_range("2023-01-01", periods=2, freq="h")
    apply_index = pd.date_range("2023-02-01", periods=3, freq="h")
    base_values = np.array([1000.123456789, 2000.25, 3000.75])
    base = pd.Series(base_values, index=apply_index, name="kpx_group_1")
    features = _features(apply_index, base_values / CAPACITY_KWH["kpx_group_1"])
    # row-major: row0 chooses greater 0.95 magnitude, row1 is identity because
    # both fail one condition, row2 exact magnitude tie chooses 0.98.
    probability = np.array([0.60, 0.60, 0.54, 0.90, 0.90, 0.90])
    magnitude = np.array([0.01, 0.02, 0.03, -0.01, 0.015, 0.015])
    policy = ActionUpliftPolicy()
    policy.classifier_ = _FakeClassifier(probability)  # type: ignore[assignment]
    policy.regressor_ = _FakeRegressor(magnitude)  # type: ignore[assignment]
    policy.fit_index_ = fit_index
    candidate, diagnostics = policy.predict(
        features, base, capacity_kwh=CAPACITY_KWH["kpx_group_1"]
    )
    np.testing.assert_array_equal(diagnostics["selected_factor"], [0.95, 1.0, 0.98])
    assert candidate.iloc[1] == base.iloc[1]
    assert np.array_equal(candidate.to_numpy()[[1]], base_values[[1]])


def test_group_gate_and_delta_transfer() -> None:
    index = pd.date_range("2024-01-01 01:00", periods=4, freq="h")
    capacity = CAPACITY_KWH["kpx_group_3"]
    actual = pd.Series(np.array([0.30, 0.40, 0.50, 0.60]) * capacity, index=index)
    baseline = pd.Series(np.array([0.34, 0.44, 0.54, 0.64]) * capacity, index=index)
    candidate = pd.Series(np.array([0.32, 0.42, 0.52, 0.62]) * capacity, index=index)
    blocks = {"full": (str(index.min()), str(index.max()))}
    comparisons = compare_group_blocks(
        actual, baseline, candidate, group="kpx_group_3", blocks=blocks
    )
    assert passes_all_blocks(comparisons, full_block="full")
    recent = baseline + 123.0
    transferred = transfer_delta(recent, baseline, candidate, capacity_kwh=capacity)
    np.testing.assert_allclose(transferred - recent, candidate - baseline)


def _local_import_closure(entry: Path) -> tuple[str, ...]:
    pending = [entry]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "scripts":
                    candidates.extend(f"scripts.{alias.name}" for alias in node.names)
                else:
                    candidates.append(node.module)
            elif isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            for module in candidates:
                if not (module.startswith("src.") or module.startswith("scripts.")):
                    continue
                candidate = PROJECT_DIR / (module.replace(".", "/") + ".py")
                if candidate.exists() and candidate not in visited:
                    pending.append(candidate)
    return tuple(
        sorted(str(path.relative_to(PROJECT_DIR)).replace("\\", "/") for path in visited)
    )


def test_static_source_closure_is_exact() -> None:
    closure = _local_import_closure(PROJECT_DIR / "scripts/run_action_uplift_policy.py")
    assert closure == tuple(sorted(runner.EXPECTED_SOURCE_CLOSURE))


def test_reader_order_locks_predictions_before_outer_and_2024_labels() -> None:
    stage1 = inspect.getsource(runner._stage1)
    assert stage1.index("_write_source_lock") < stage1.index("_load_stage1_sources")
    assert stage1.index("_write_source_lock") < stage1.index("_read_stage1_weather")
    assert stage1.index("_write_source_lock") < stage1.index("labels_g12, labels_g12_evidence = _read_label_prefix")
    assert stage1.index("prescore_path = out_dir / \"stage1_prescore_lock.json\"") < stage1.index(
        "outer_labels, outer_evidence = _read_label_prefix"
    )
    stage2 = inspect.getsource(runner._stage2)
    assert stage2.index("if not passed_groups") < stage2.index("_load_stage1_sources")
    assert stage2.index("prescore_path = out_dir / \"stage2_prescore_lock.json\"") < stage2.index(
        "labels, label_evidence = _read_full_labels"
    )


def test_no_candidate_stage2_and_final_do_not_open_conditional_inputs(tmp_path: Path) -> None:
    stage1_lock = {"passed_groups": []}
    runner._write_json(tmp_path / "stage1_promotion_lock.json", stage1_lock)
    stage2_lock = runner._stage2_no_candidate(tmp_path, stage1_lock)
    assert stage2_lock["promoted"] is False
    result = json.loads((tmp_path / "stage2_results.json").read_text(encoding="utf-8"))
    assert result["2024_component_read"] is False
    assert result["2024_label_read"] is False
    assert result["recent_v4_read"] is False
    final = runner._final_no_candidate(tmp_path, stage2_lock)
    assert final["2025_components_read"] is False
    assert final["2025_recent_v4_read"] is False
    assert final["csv_created"] is False

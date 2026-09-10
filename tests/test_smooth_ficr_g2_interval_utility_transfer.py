from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_smooth_ficr_g2_interval_utility_transfer as runner
from src.metric import CAPACITY_KWH, TARGET_COLS


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_hash_and_rule() -> None:
    assert runner.sha256_file(runner.CONFIG_PATH) == runner.EXPECTED_CONFIG_SHA
    assert runner.CONFIG_SIDECAR.read_text(encoding="utf-8").strip() == runner.EXPECTED_CONFIG_SHA
    config = json.loads(runner.CONFIG_PATH.read_text(encoding="utf-8"))
    rule = config["immutable_single_candidate"]
    assert rule["active_group"] == "kpx_group_2"
    assert rule["identity_groups"] == ["kpx_group_1", "kpx_group_3"]
    assert rule["blend_weight"] == 0.025
    assert rule["utility_advantage_margin"] == 0.01
    assert rule["recent_v4_independent_gate_forbidden"] is True
    risk = config["risk_classification"]
    assert risk["result_informed_G2_selection"] is True
    assert risk["multiple_testing"] is True
    assert risk["selection_unsafe"] is True
    assert risk["private_champion"] is False


def test_public_scale_paths_are_absent_from_runner_inputs() -> None:
    config = json.loads(runner.CONFIG_PATH.read_text(encoding="utf-8"))
    paths = [str(spec["path"]).lower() for spec in runner._iter_specs(config)]
    assert not any("public_scale" in path or "scale_09" in path for path in paths)
    exclusion = config["public_and_scale_exclusion"]
    assert exclusion["no_public_or_scale_path_may_be_read_by_the_runner"] is True


def test_default_namespace_is_postrun_safe() -> None:
    args = runner.parse_args([])
    assert args.stage == "all"
    assert args.out_dir.as_posix().endswith(
        "artifacts/postgate/smooth_ficr_g2_interval_utility_transfer_v1"
    )
    assert args.config.as_posix().endswith(
        "configs/smooth_ficr_g2_interval_utility_transfer_preregister_v1.json"
    )


def test_verifier_accepts_locked_output_size_bytes_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "locked.bin"
    path.write_bytes(b"locked-output")
    descriptor = runner.describe_file(path)
    assert "size_bytes" in descriptor and "bytes" not in descriptor
    assert runner._verify(descriptor) == path


def test_row_linear_interpolation() -> None:
    surface = np.tile(np.arange(103, dtype=np.float64), (3, 1))
    query = np.array([0.0, 0.125, 1.02], dtype=np.float64)
    observed = runner.interpolated_utility(surface, query)
    np.testing.assert_allclose(observed, [0.0, 12.5, 102.0], rtol=0.0, atol=0.0)


def _frame(index: pd.DatetimeIndex, g2_cf: float) -> pd.DataFrame:
    values = {
        "kpx_group_1": np.full(len(index), 0.4 * CAPACITY_KWH["kpx_group_1"]),
        "kpx_group_2": np.full(len(index), g2_cf * CAPACITY_KWH["kpx_group_2"]),
        "kpx_group_3": np.full(len(index), 0.5 * CAPACITY_KWH["kpx_group_3"]),
    }
    return pd.DataFrame(values, index=index, columns=TARGET_COLS, dtype=np.float64)


def test_primary_gate_delta_is_transferred_once_and_identity_bits_hold() -> None:
    index = pd.date_range("2024-01-01 01:00", periods=8, freq="h", name="forecast_kst_dtm")
    primary = _frame(index, 0.4)
    recent = _frame(index, 0.42)
    raw = pd.DataFrame(0.0, index=index, columns=TARGET_COLS, dtype=np.float64)
    raw.loc[:, "kpx_group_2"] = 1.0
    utility = np.tile(np.arange(103, dtype=np.float64) / 10.0, (len(index), 1))
    p_candidate, r_candidate, delta, diagnostics = runner.compose_primary_delta(
        primary, recent, raw, utility
    )
    assert diagnostics["gate"].all()
    np.testing.assert_array_equal(
        r_candidate["kpx_group_2"].to_numpy() - recent["kpx_group_2"].to_numpy(),
        delta["kpx_group_2"].to_numpy(),
    )
    np.testing.assert_array_equal(
        p_candidate["kpx_group_2"].to_numpy() - primary["kpx_group_2"].to_numpy(),
        delta["kpx_group_2"].to_numpy(),
    )
    for group in runner.IDENTITY_GROUPS:
        assert p_candidate[group].to_numpy().tobytes() == primary[group].to_numpy().tobytes()
        assert r_candidate[group].to_numpy().tobytes() == recent[group].to_numpy().tobytes()


def test_margin_is_strictly_greater_than() -> None:
    index = pd.date_range("2024-01-01 01:00", periods=2, freq="h", name="forecast_kst_dtm")
    primary = _frame(index, 0.4)
    recent = _frame(index, 0.4)
    raw = pd.DataFrame(0.4, index=index, columns=TARGET_COLS, dtype=np.float64)
    utility = np.zeros((len(index), 103), dtype=np.float64)
    p_candidate, r_candidate, delta, diagnostics = runner.compose_primary_delta(
        primary, recent, raw, utility
    )
    assert not diagnostics["gate"].any()
    assert not np.count_nonzero(delta.to_numpy())
    assert p_candidate.to_numpy().tobytes() == primary.to_numpy().tobytes()
    assert r_candidate.to_numpy().tobytes() == recent.to_numpy().tobytes()


def test_registered_7_plus_7_score_gate() -> None:
    index = runner._segments(2024)["full"]
    capacity = float(CAPACITY_KWH[runner.GROUP])
    actual = pd.Series(0.5 * capacity, index=index)
    baseline = pd.Series(0.59 * capacity, index=index)
    candidate = actual.copy()
    result = runner.score_g2_identity_mix(actual, baseline, candidate, year=2024)
    assert result["G2_7_of_7_strict_positive"] is True
    assert result["mixed_7_of_7_strict_positive"] is True
    assert result["full_G2_and_mixed_components_nonnegative"] is True
    assert result["registered_strict_deltas"] == 14
    assert result["passed"] is True
    for record in result["comparisons"].values():
        for key, value in record["delta"].items():
            assert record["mixed_delta_by_exact_identity_additivity"][key] == value / 3.0


def test_recursive_closure_contains_runner_test_and_models() -> None:
    args = runner.parse_args([])
    closure = runner._closure(args)
    paths = {Path(item["path"]).resolve() for item in closure["resolved_files"]}
    assert Path(runner.__file__).resolve() in paths
    assert Path(__file__).resolve() in paths
    assert (ROOT / "src/raw_spatiotemporal_smooth_ficr.py").resolve() in paths
    assert (ROOT / "src/direct_interval_probability.py").resolve() in paths


def test_canonical_if_present_is_bound_to_same_preregister() -> None:
    manifest_path = runner.DEFAULT_OUT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["config_sha256"] == runner.EXPECTED_CONFIG_SHA
        assert manifest["artifact_type"] == "smooth_ficr_g2_interval_utility_transfer_v1"
        assert manifest["no_rule_change_retune_retry_rescue_or_alternative"] is True

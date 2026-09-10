from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_gfs_global_revision_paired_increment as runner


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _external_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        ["2024-07-01 00:00", "2024-07-01 01:00", "2024-07-01 13:00", "2024-07-01 14:00"],
        name="forecast_kst_dtm",
    )
    values: dict[str, np.ndarray] = {}
    for height in runner.HEIGHTS:
        values[f"wind_speed_{height}m_previous_day1"] = np.array([1, 2, 3, 4], dtype=float) + height
        values[f"wind_direction_{height}m_previous_day1"] = np.array([0, 90, 180, 270], dtype=float)
        values[f"wind_speed_{height}m_previous_day2"] = np.array([10, 20, 30, 40], dtype=float) + height
        values[f"wind_direction_{height}m_previous_day2"] = np.array([0, 90, 180, 270], dtype=float)
    return pd.DataFrame(values, index=index)


def test_preregistration_hash_sidecar_and_fixed_contract() -> None:
    assert _sha(runner.CONFIG_PATH) == runner.CONFIG_SHA
    assert runner.SIDECAR_PATH.read_text(encoding="utf-8") == (
        f"{runner.CONFIG_SHA}  {runner.CONFIG_PATH.name}\n"
    )
    config = runner.verify_config(runner.CONFIG_PATH)
    assert config["paired_model"]["transfer_weight"] == 0.25
    assert config["paired_model"]["feature_count"] == 12
    assert config["risk_and_scope"]["candidate_predictions_or_scores_before_this_lock"] == 0
    assert config["risk_and_scope"]["year_2025_external_requests_response_bytes_or_values"] == 0


def test_candidate_A_B_hour_masks_are_exact_and_hour14_day1_forbidden() -> None:
    source = _external_frame()
    a = runner.select_external_features(source, runner.CANDIDATES[0])
    b = runner.select_external_features(source, runner.CANDIDATES[1])
    for height in runner.HEIGHTS:
        assert a[f"gfsrev__ws{height}_ms"].tolist() == [10 + height, 20 + height, 30 + height, 40 + height]
        assert b[f"gfsrev__ws{height}_ms"].tolist() == [10 + height, 2 + height, 3 + height, 40 + height]


def test_direction_conversion_and_twelve_feature_order() -> None:
    source = _external_frame()
    selected = runner.select_external_features(source, runner.CANDIDATES[0])
    control = pd.DataFrame({"cross__hub_ws_mean": [1.0, 2.0, 3.0, 4.0]}, index=source.index)
    extension = runner.add_disagreements(selected, control)
    assert tuple(extension.columns) == runner.EXTENDED_COLUMNS
    assert len(extension.columns) == 12
    for height in runner.HEIGHTS:
        speed = source[f"wind_speed_{height}m_previous_day2"].to_numpy()
        np.testing.assert_allclose(extension[f"gfsrev__u{height}_ms"], [0, -speed[1], 0, speed[3]], atol=1e-5)
        np.testing.assert_allclose(extension[f"gfsrev__v{height}_ms"], [-speed[0], 0, speed[2], 0], atol=1e-5)
        np.testing.assert_allclose(
            extension[f"gfsrev__ws{height}_minus_cross_hub_ws_mean"], speed - np.array([1, 2, 3, 4])
        )


def test_terminal_identity_is_float64_bit_exact() -> None:
    index = pd.DatetimeIndex(["2024-12-31 23:00", "2025-01-01 00:00"], name="forecast_kst_dtm")
    baseline = pd.Series([1234.567890123, 9876.543210987], index=index)
    increment = pd.Series([0.01, 0.0], index=index)
    result = runner.apply_increment(baseline, increment, capacity=21600.0)
    assert np.float64(result.iloc[-1]).tobytes() == np.float64(baseline.iloc[-1]).tobytes()


def test_regime_thresholds_are_exact_and_cover_rows() -> None:
    values = pd.Series([0.0, 3.999999, 4.0, 7.999999, 8.0, 20.0])
    assert runner.regime_name(values).tolist() == ["low", "low", "mid", "mid", "high", "high"]


def _passing_comparisons() -> dict[str, object]:
    def record(delta: float = 0.001) -> dict[str, float]:
        return {"delta_total_score": delta, "delta_one_minus_nmae": 0.001, "delta_ficr": 0.001}

    payload: dict[str, object] = {}
    for baseline in runner.BASELINES:
        payload[baseline] = {
            "groups": {
                group: {
                    "time": {name: record() for name in runner.TIME_SLICES},
                    "regime": {name: record() for name in runner.REGIMES},
                }
                for group in runner.TARGET_COLS
            },
            "mixed": {
                "time": {name: record() for name in runner.TIME_SLICES},
                "regime": {name: record() for name in runner.REGIMES},
            },
        }
    return payload


def test_gate_requires_all_96_dual_baseline_records_strictly_positive() -> None:
    comparisons = _passing_comparisons()
    result = runner.gate_summary(comparisons)
    assert result["dual_gate_pass"] is True
    assert result["registered_group_records_across_both_baselines"] == 72
    comparisons["primary_corrected_v3"]["groups"]["kpx_group_1"]["regime"]["low"]["delta_total_score"] = 0.0
    assert runner.gate_summary(comparisons)["dual_gate_pass"] is False


def test_gate_requires_H2_mixed_components_nonnegative() -> None:
    comparisons = _passing_comparisons()
    comparisons["interaction_recent_v4"]["mixed"]["time"]["H2"]["delta_ficr"] = -1e-15
    assert runner.gate_summary(comparisons)["dual_gate_pass"] is False


def test_physical_source_is_2024_only_and_complete_on_fit_apply() -> None:
    config = runner.verify_config(runner.CONFIG_PATH)
    runner.validate_inputs(config)
    frame = runner.read_external()
    assert len(frame) == 26352
    assert frame["time"].min() == pd.Timestamp("2024-01-01 00:00")
    assert frame["time"].max() == pd.Timestamp("2024-12-31 23:00")
    variables = frame.columns[2:]
    for group in runner.TARGET_COLS:
        local = frame.loc[frame["group"] == group].set_index("time")
        assert local.loc[runner.FIT_INDEX, variables].notna().all().all()
        assert local.loc[runner.MODEL_APPLY_INDEX, variables].notna().all().all()


def test_recursive_AST_closure_contains_runner_metric_and_no_unresolved() -> None:
    closure = runner.resolve_ast_closure(ROOT / "scripts/run_gfs_global_revision_paired_increment.py")
    relative = {path.relative_to(ROOT).as_posix() for path in closure}
    assert "scripts/run_gfs_global_revision_paired_increment.py" in relative
    assert "src/metric.py" in relative
    assert "src/__init__.py" in relative


def test_phase_order_locks_candidate_before_score_label_reader_and_metric() -> None:
    source = inspect.getsource(runner.main)
    assert source.index("candidate_lock_path") < source.index("read_score_labels(")
    assert source.index("score_lock_path") < source.index("comparison_payload(")
    assert source.index("read_fit_labels_only()") < source.index("control_model.fit(")


def test_source_lock_declares_zero_label_content_before_fit() -> None:
    source = inspect.getsource(runner.main)
    assert source.index('"label_content_bytes_read": 0') < source.index("read_fit_labels_only()")
    closure_source = inspect.getsource(runner.source_closure)
    assert '"content_bytes_read_for_this_lock": 0' in closure_source


def test_existing_output_rejected_before_any_label_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "exists"
    out.mkdir()
    reached = False

    def forbidden() -> object:
        nonlocal reached
        reached = True
        raise AssertionError("label reader reached")

    monkeypatch.setattr(runner, "read_fit_labels_only", forbidden)
    with pytest.raises(FileExistsError):
        runner.main(["--out-dir", str(out)])
    assert reached is False


def test_shared_guard_is_exclusive_and_owner_release_only(tmp_path: Path) -> None:
    path = tmp_path / "guard.json"
    owner = runner.acquire_heavy_guard(path)
    assert owner["pid"] == os.getpid()
    with pytest.raises(RuntimeError, match="held by live PID"):
        runner.acquire_heavy_guard(path)
    runner.release_heavy_guard(path, {**owner, "token": "wrong"})
    assert path.exists()
    runner.release_heavy_guard(path, owner)
    assert not path.exists()


def test_time_slice_counts_and_terminal_contract() -> None:
    counts = {
        name: int(((runner.SCORE_INDEX >= start) & (runner.SCORE_INDEX <= end)).sum())
        for name, (start, end) in runner.TIME_SLICES.items()
    }
    assert counts == {
        "H2": 4417,
        "Q3": 2208,
        "Q4": 2209,
        "Jul": 744,
        "Aug": 744,
        "Sep": 720,
        "Oct": 744,
        "Nov": 720,
        "Dec": 745,
    }
    assert runner.SCORE_INDEX[-1] == runner.TERMINAL


def test_no_csv_and_2025_external_paths_are_hard_blocked_by_contract() -> None:
    config = json.loads(runner.CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["execution_contract"]["on_failure"].startswith("reject and manifest; no CSV")
    manifest = json.loads(runner.SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["year_2025_requests_response_bytes_or_values"] == 0
    assert manifest["calendar_years_requested_or_parsed"] == [2024]

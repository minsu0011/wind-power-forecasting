from __future__ import annotations

import json
import os
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_icon_global_paired_increment as icon_runner
from scripts.download_openmeteo_icon_global_previous_runs import query_url
from scripts.run_icon_global_paired_increment import (
    CANDIDATES,
    EXTENDED_COLUMNS,
    acquire_heavy_guard,
    add_disagreements,
    apply_paired_increment,
    release_heavy_guard,
    select_external_features,
)


def external_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        ["2024-07-01 00:00", "2024-07-01 01:00", "2024-07-01 13:00", "2024-07-01 14:00"],
        name="forecast_kst_dtm",
    )
    data: dict[str, list[float]] = {}
    for height, factor in ((10, 1.0), (80, 2.0), (120, 3.0)):
        data[f"wind_speed_{height}m_previous_day1"] = [factor, 2 * factor, 3 * factor, 4 * factor]
        data[f"wind_direction_{height}m_previous_day1"] = [0.0, 90.0, 180.0, 270.0]
        data[f"wind_speed_{height}m_previous_day2"] = [10 * factor, 20 * factor, 30 * factor, 40 * factor]
        data[f"wind_direction_{height}m_previous_day2"] = [0.0, 90.0, 180.0, 270.0]
    return pd.DataFrame(data, index=index)


def test_candidate_masks_are_exact_at_all_heights_and_hour14_day1_is_forbidden() -> None:
    frame = external_frame()
    day2 = select_external_features(frame, CANDIDATES[0])
    composite = select_external_features(frame, CANDIDATES[1])
    for height, factor in ((10, 1.0), (80, 2.0), (120, 3.0)):
        assert day2[f"icon__ws{height}_ms"].tolist() == [10 * factor, 20 * factor, 30 * factor, 40 * factor]
        assert composite[f"icon__ws{height}_ms"].tolist() == [10 * factor, 2 * factor, 3 * factor, 40 * factor]


def test_exact_twelve_features_direction_conversion_and_disagreement() -> None:
    selected = select_external_features(external_frame(), CANDIDATES[0])
    control = pd.DataFrame({"cross__hub_ws_mean": [1.0, 2.0, 3.0, 4.0]}, index=selected.index)
    features = add_disagreements(selected, control)
    assert tuple(features.columns) == EXTENDED_COLUMNS
    assert len(features.columns) == 12
    assert not any("shear" in name or "grid" in name for name in features.columns)
    np.testing.assert_allclose(features["icon__u10_ms"], [0.0, -20.0, 0.0, 40.0], atol=1e-5)
    np.testing.assert_allclose(features["icon__v10_ms"], [-10.0, 0.0, 30.0, 0.0], atol=1e-5)
    assert features["icon__ws10_minus_cross_hub_ws_mean"].tolist() == [9.0, 18.0, 27.0, 36.0]


def test_zero_terminal_increment_preserves_baseline_bit_identity() -> None:
    index = pd.DatetimeIndex(["2024-12-31 23:00", "2025-01-01 00:00"])
    baseline = pd.Series([1234.567890123, 9876.543210987], index=index)
    increment = pd.Series([0.01, 0.0], index=index)
    candidate = apply_paired_increment(baseline, increment, capacity_kwh=21_600.0)
    assert np.float64(candidate.iloc[-1]).tobytes() == np.float64(baseline.iloc[-1]).tobytes()


def test_downloader_has_physical_2025_network_block() -> None:
    with pytest.raises(AssertionError, match="only calendar 2022--2024"):
        query_url(37.28, 128.96, start="2025-01-01", end="2025-01-01")


def test_heavy_guard_is_exclusive_and_owner_only_release(tmp_path: Path) -> None:
    path = tmp_path / "heavy_cpu_fit.pid.json"
    owner = acquire_heavy_guard(path)
    assert owner["pid"] == os.getpid()
    assert owner["experiment_id"] == "icon_global_paired_increment_2024_forward_v3"
    assert len(owner["token"]) == 32
    with pytest.raises(RuntimeError, match="held by live PID"):
        acquire_heavy_guard(path)
    release_heavy_guard(path, {"pid": owner["pid"], "token": "wrong"})
    assert path.exists()
    release_heavy_guard(path, owner)
    assert not path.exists()


def test_heavy_guard_reclaims_confirmed_stale_pid(tmp_path: Path) -> None:
    path = tmp_path / "heavy_cpu_fit.pid.json"
    path.write_text(json.dumps({"pid": 2_147_483_647}), encoding="utf-8")
    owner = acquire_heavy_guard(path)
    assert owner["pid"] == os.getpid()
    release_heavy_guard(path, owner)


def test_frozen_preregistration_exact_candidate_and_gate_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    base_path = root / "configs/icon_global_paired_increment_preregister_v1.json"
    phase_path = root / "configs/icon_global_paired_increment_preregister_v2.json"
    path = root / "configs/icon_global_paired_increment_preregister_v3.json"
    sidecar = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sidecar
    base_sidecar = base_path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == base_sidecar
    phase_sidecar = phase_path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(phase_path.read_bytes()).hexdigest() == phase_sidecar
    payload = json.loads(path.read_text(encoding="utf-8"))
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    assert payload["base_preregister"]["sha256"] == phase_sidecar
    assert phase["base_preregister"]["sha256"] == base_sidecar
    assert payload["reaffirmed"]["candidate_ids_fixed_order"] == list(CANDIDATES)
    assert base["paired_model"]["transfer_weight"] == 0.25
    assert base["paired_model"]["extended_features_after_control"] == list(EXTENDED_COLUMNS)
    assert base["paired_model"]["parameters"]["objective"] == "regression_l1"
    assert phase["label_access_contract"]["h2_label_bytes_or_value_cells_read_before_candidate_lock"] == 0
    assert phase["guard_contract"]["owner_experiment_id"] == "icon_global_paired_increment_2024_forward_v2"
    assert payload["only_change"]["candidate_or_statistical_contract_change"] is False


def test_two_phase_label_reader_stops_at_exact_H1_offset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "train_labels.csv"
    full_index = pd.date_range("2022-01-01 01:00", "2025-01-01 00:00", freq="h")
    lines = ["kst_dtm,kpx_group_1,kpx_group_2,kpx_group_3\n"]
    lines.extend(f"{timestamp:%Y-%m-%d %H:%M:%S},1,2,3\n" for timestamp in full_index)
    path.write_text("".join(lines), encoding="utf-8")
    monkeypatch.setattr(icon_runner, "LABEL_PATH", path)
    monkeypatch.setattr(icon_runner, "EXPECTED_LABEL_BYTES", path.stat().st_size)
    fit, offset, fit_record, header = icon_runner.read_fit_labels_only()
    assert len(fit) == len(pd.date_range("2024-02-18 00:00", "2024-06-30 23:00", freq="h"))
    assert fit_record["H2_bytes_or_value_cells_read"] == 0
    assert 0 < offset < path.stat().st_size
    score, score_record = icon_runner.read_score_labels(offset, header)
    assert len(score) == 4417
    assert score.index[0] == pd.Timestamp("2024-07-01 00:00")
    assert score.index[-1] == pd.Timestamp("2025-01-01 00:00")
    assert score_record["bytes_or_rows_after_segment_read"] == 0


def test_main_phase_order_locks_candidates_before_H2_label_read_and_metric() -> None:
    source = inspect.getsource(icon_runner.main)
    markers = [
        "closure_path = source_lock",
        "guard_owner = acquire_heavy_guard",
        "read_fit_labels_only()",
        "fit_lock_path = fit_label_lock",
        "control_model.fit",
        "candidate_lock_path = candidate_lock",
        "read_score_labels(score_offset, label_header)",
        "comparison_payload(labels_apply",
    ]
    positions = [source.index(marker) for marker in markers]
    assert positions == sorted(positions)

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_full_weather_ordinal_bayes_stage2_final as runner
from src.metric import TARGET_COLS


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_stage1_manifest_and_g3_promotion_verify() -> None:
    manifest, result = runner._verify_stage1(
        ROOT / "artifacts/postgate/full_weather_ordinal_bayes_strict_v1"
    )
    assert manifest["status"] == "stage1_group_promotion_pending_stage2"
    assert result["passed_groups"] == ["kpx_group_3"]
    assert result["year_2024_value_bytes_read"] == 0
    assert result["csv_files_created"] == 0


def test_default_append_only_namespace() -> None:
    args = runner.parse_args([])
    assert args.out_dir.as_posix().endswith(
        "artifacts/postgate/full_weather_ordinal_bayes_strict_v1"
    )
    assert args.preregister.name == "full_weather_ordinal_bayes_preregister_v1.json"
    assert runner.TRANSFER_WEIGHT == 0.10
    assert runner.G3 == "kpx_group_3"


def test_same_primary_delta_transfer_and_g1g2_bits() -> None:
    index = pd.date_range("2024-01-01 01:00", periods=4, freq="h")
    primary = pd.DataFrame(
        {
            "kpx_group_1": [1.0, 2.0, 3.0, 4.0],
            "kpx_group_2": [5.0, 6.0, 7.0, 8.0],
            "kpx_group_3": np.array([0.20, 0.40, 0.80, 1.00]) * 21000.0,
        },
        index=index,
    )
    recent = primary.copy()
    recent["kpx_group_1"] += 100.0
    recent["kpx_group_2"] += 200.0
    recent["kpx_group_3"] = np.array([0.21, 0.39, 0.78, 0.99]) * 21000.0
    action = pd.Series([0.30, 0.30, 0.95, 1.02], index=index)
    candidate_primary, candidate_recent, delta = runner._same_delta_candidate(
        primary, recent, action
    )
    expected_primary_cf = 0.9 * primary["kpx_group_3"].to_numpy() / 21000.0 + 0.1 * action.to_numpy()
    expected_delta = expected_primary_cf - primary["kpx_group_3"].to_numpy() / 21000.0
    np.testing.assert_allclose(delta.to_numpy(), expected_delta, rtol=0, atol=1e-15)
    expected_recent = np.clip(
        recent["kpx_group_3"].to_numpy() / 21000.0 + expected_delta, 0.0, 1.02
    ) * 21000.0
    np.testing.assert_allclose(
        candidate_recent["kpx_group_3"], expected_recent, rtol=0, atol=1e-12
    )
    for group in TARGET_COLS[:2]:
        assert candidate_primary[group].to_numpy().tobytes() == primary[group].to_numpy().tobytes()
        assert candidate_recent[group].to_numpy().tobytes() == recent[group].to_numpy().tobytes()


def _record(delta: float, n: float = 0.001, f: float = 0.001) -> dict:
    return {
        "baseline": {"score": 0.5, "total_score": 0.5, "one_minus_nmae": 0.8, "ficr": 0.2},
        "candidate": {"score": 0.5 + delta, "total_score": 0.5 + delta, "one_minus_nmae": 0.8 + n, "ficr": 0.2 + f},
        "delta": delta,
    }


def test_dual_gate_requires_group_mixed_all7_and_components(monkeypatch) -> None:
    index = pd.DatetimeIndex([pd.Timestamp("2024-01-01 01:00")])
    segments = {name: index for name in runner.REQUIRED_SEGMENTS}
    monkeypatch.setattr(runner.bounded, "_year_segments", lambda year: segments)
    group_records = {name: _record(0.001) for name in runner.REQUIRED_SEGMENTS}
    mixed_records = {name: _record(0.001) for name in runner.REQUIRED_SEGMENTS}
    monkeypatch.setattr(runner.strict, "_comparison", lambda *args, **kwargs: group_records)
    monkeypatch.setattr(runner, "_mixed_comparison", lambda *args, **kwargs: mixed_records)
    labels = pd.DataFrame([[1.0, 1.0, 1.0]], index=index, columns=TARGET_COLS)
    bases = {name: labels.copy() for name in ("primary_v3", "recent_v4")}
    candidates = {name: labels.copy() for name in bases}
    passed, audit = runner._dual_gate(labels, bases, candidates, 2024)
    assert passed
    assert all(audit[name]["gate"]["passed"] for name in audit)
    mixed_records["Q2"] = _record(0.0)
    assert not runner._dual_gate(labels, bases, candidates, 2024)[0]


def test_reader_order_has_dual_candidate_lock_before_2024_labels() -> None:
    source = inspect.getsource(runner._stage2)
    lock_index = source.index('prescore_lock = args.out_dir / "stage2_2024_dual_prescore_lock.json"')
    label_index = source.index("full_labels = strict._read_full_labels(label_path)")
    assert lock_index < label_index
    assert '"created_before_any_2024_application_label_value": True' in source[:label_index]
    assert '"recent_uses_exact_primary_delta": True' in source[:label_index]
    assert '"g1_g2_bit_identity": True' in source[:label_index]


def test_2025_inputs_are_only_snapshotted_after_stage2_promotion() -> None:
    stage2_source = inspect.getsource(runner._stage2_input_snapshot)
    assert "sample_submission" not in stage2_source
    assert "weather_test" not in stage2_source
    assert "final_primary" not in stage2_source
    run_source = inspect.getsource(runner.run)
    conditional = run_source.index("if promoted:")
    final_snapshot = run_source.index("final_input_before = _final_input_snapshot(args)")
    final_call = run_source.index("final_result = _final")
    assert conditional < final_snapshot < final_call


def test_final_csv_is_after_prescore_lock_and_pass_only() -> None:
    final_source = inspect.getsource(runner._final)
    assert final_source.index('prescore = args.out_dir / "final_2025_prescore_lock.json"') < final_source.index(
        "strict._atomic_csv(submission, csv_path)"
    )
    run_source = inspect.getsource(runner.run)
    assert "if promoted:" in run_source
    assert "final_result = _final" in run_source


def test_no_public_or_scale_input_paths() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8").lower()
    assert "public_scale_probe" not in source
    assert "scale_095" not in source
    assert "public_feedback" not in source
    payload = json.loads(
        (ROOT / "configs/full_weather_ordinal_bayes_preregister_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["public_exclusion"]["public_feedback_used"] is False

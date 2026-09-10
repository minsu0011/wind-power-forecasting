from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from scripts import run_direct_interval_scale097_margin0 as runner


def _baseline() -> pd.DataFrame:
    index = pd.date_range("2023-01-01 01:00", periods=3, freq="h", name="forecast_kst_dtm")
    return pd.DataFrame(
        {
            "kpx_group_1": np.array([0.20, 0.50, 0.80]) * 21600.0,
            "kpx_group_2": np.array([0.30, 0.60, 0.90]) * 21600.0,
            "kpx_group_3": np.array([0.25, 0.55, 0.85]) * 21000.0,
        },
        index=index,
    )


def test_margin_zero_is_strict_and_flat_utility_is_identity() -> None:
    baseline = _baseline()
    flat = np.ones((len(baseline), 103), dtype=np.float64)
    candidate, diagnostics = runner.apply_rule(
        baseline, {group: flat.copy() for group in runner.ACTIVE_GROUPS}
    )
    for group in runner.ACTIVE_GROUPS:
        assert diagnostics[group]["gate"].tolist() == [False, False, False]
        np.testing.assert_array_equal(
            candidate[group].to_numpy().view(np.uint64),
            baseline[group].to_numpy().view(np.uint64),
        )


def test_decreasing_utility_gates_exact_097_scale() -> None:
    baseline = _baseline()
    grid = np.arange(103, dtype=np.float64) / 100.0
    decreasing = np.tile(-grid, (len(baseline), 1))
    candidate, diagnostics = runner.apply_rule(
        baseline, {group: decreasing.copy() for group in runner.ACTIVE_GROUPS}
    )
    for group in runner.ACTIVE_GROUPS:
        assert diagnostics[group]["gate"].all()
        np.testing.assert_array_equal(
            candidate[group].to_numpy(), baseline[group].to_numpy() * 0.97
        )


def test_g3_is_always_bit_exact_identity() -> None:
    baseline = _baseline()
    rng = np.random.default_rng(42)
    utility = {group: rng.normal(size=(len(baseline), 103)) for group in runner.ACTIVE_GROUPS}
    candidate, _ = runner.apply_rule(baseline, utility)
    np.testing.assert_array_equal(
        candidate[runner.IDENTITY_GROUP].to_numpy().view(np.uint64),
        baseline[runner.IDENTITY_GROUP].to_numpy().view(np.uint64),
    )


def test_operating_year_segment_contract() -> None:
    segments = runner._segments(2024)
    assert len(segments["full"]) == 8784
    assert len(segments["H1"]) == 4368
    assert len(segments["H2"]) == 4416
    assert [len(segments[f"Q{i}"]) for i in range(1, 5)] == [2184, 2184, 2208, 2208]
    combined = segments["Q1"].append([segments["Q2"], segments["Q3"], segments["Q4"]])
    assert combined.equals(segments["full"])


def test_default_namespace_and_config_hash_are_frozen() -> None:
    args = runner.parse_args([])
    assert args.out_dir.as_posix() == "artifacts/postgate/direct_interval_scale097_margin0_v1"
    assert hashlib.sha256(Path(args.config).read_bytes()).hexdigest() == runner.CONFIG_SHA256
    assert float(runner.FACTOR) == 0.97
    assert float(runner.MARGIN) == 0.0


def test_existing_output_directory_is_rejected() -> None:
    args = runner.parse_args([])
    with tempfile.TemporaryDirectory() as temporary:
        args.out_dir = Path(temporary)
        with pytest.raises(FileExistsError, match="existing output directory"):
            runner.run(args)

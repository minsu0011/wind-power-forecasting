from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from scripts import run_direct_interval_scale097_margin001 as runner


def _baseline() -> pd.DataFrame:
    index = pd.date_range("2023-01-01 01:00", periods=2, freq="h", name="forecast_kst_dtm")
    return pd.DataFrame(
        {
            "kpx_group_1": np.array([0.4, 0.8]) * 21600.0,
            "kpx_group_2": np.array([0.5, 0.9]) * 21600.0,
            "kpx_group_3": np.array([0.6, 0.7]) * 21000.0,
        },
        index=index,
    )


def test_advantage_equal_margin_is_identity() -> None:
    baseline = _baseline()
    baseline.loc[baseline.index[0], "kpx_group_1"] = 21600.0
    utility = np.zeros((len(baseline), 103), dtype=np.float64)
    # Both queries are exact grid points: U(.97)-U(1.00)=.01 exactly.
    utility[0, 97] = np.float64(0.01)
    candidate, diagnostics = runner.apply_rule(
        baseline, {group: utility.copy() for group in runner.ACTIVE_GROUPS}
    )
    assert not bool(diagnostics["kpx_group_1"]["gate"].iloc[0])
    assert candidate["kpx_group_1"].iloc[0] == baseline["kpx_group_1"].iloc[0]


def test_advantage_above_margin_uses_exact_097() -> None:
    baseline = _baseline()
    grid = np.arange(103, dtype=np.float64) / 100.0
    utility = np.tile(-grid, (len(baseline), 1))
    candidate, diagnostics = runner.apply_rule(
        baseline, {group: utility.copy() for group in runner.ACTIVE_GROUPS}
    )
    assert diagnostics["kpx_group_1"]["gate"].tolist() == [True, True]
    np.testing.assert_array_equal(
        candidate["kpx_group_1"].to_numpy(), baseline["kpx_group_1"].to_numpy() * 0.97
    )


def test_g3_is_bit_exact_identity() -> None:
    baseline = _baseline()
    rng = np.random.default_rng(9)
    utility = {group: rng.normal(size=(len(baseline), 103)) for group in runner.ACTIVE_GROUPS}
    candidate, _ = runner.apply_rule(baseline, utility)
    np.testing.assert_array_equal(
        candidate[runner.IDENTITY_GROUP].to_numpy().view(np.uint64),
        baseline[runner.IDENTITY_GROUP].to_numpy().view(np.uint64),
    )


def test_default_namespace_config_factor_and_margin_are_frozen() -> None:
    args = runner.parse_args([])
    assert args.out_dir.as_posix() == "artifacts/postgate/direct_interval_scale097_margin001_v1"
    assert hashlib.sha256(Path(args.config).read_bytes()).hexdigest() == runner.CONFIG_SHA256
    assert float(runner.FACTOR) == 0.97
    assert float(runner.MARGIN) == 0.01


def test_existing_output_directory_is_rejected() -> None:
    args = runner.parse_args([])
    with tempfile.TemporaryDirectory() as temporary:
        args.out_dir = Path(temporary)
        with pytest.raises(FileExistsError, match="existing output directory"):
            runner.run(args)

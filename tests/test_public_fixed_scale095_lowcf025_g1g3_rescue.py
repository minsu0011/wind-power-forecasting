from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from scripts import run_public_fixed_scale095_lowcf025_g1g3_rescue as runner


def _frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01 01:00", periods=4, freq="h", name="forecast_kst_dtm")
    return pd.DataFrame(
        {
            "kpx_group_1": np.array([0.10, 0.25, 0.251, 0.80]) * 21600.0,
            "kpx_group_2": np.array([0.10, 0.25, 0.251, 0.80]) * 21600.0,
            "kpx_group_3": np.array([0.10, 0.25, 0.251, 0.80]) * 21000.0,
        },
        index=index,
    )


def test_rule_scales_g1_g3_at_inclusive_boundary() -> None:
    baseline = _frame()
    candidate, mask = runner.apply_rescue(baseline)
    for group in runner.ACTIVE_GROUPS:
        assert mask[group].tolist() == [True, True, False, False]
        np.testing.assert_array_equal(
            candidate.loc[mask[group], group].to_numpy(),
            baseline.loc[mask[group], group].to_numpy() * 0.95,
        )


def test_g2_and_ungated_rows_are_bit_exact() -> None:
    baseline = _frame()
    candidate, mask = runner.apply_rescue(baseline)
    np.testing.assert_array_equal(
        candidate[runner.IDENTITY_GROUP].to_numpy().view(np.uint64),
        baseline[runner.IDENTITY_GROUP].to_numpy().view(np.uint64),
    )
    for group in runner.ACTIVE_GROUPS:
        identity = ~mask[group].to_numpy()
        np.testing.assert_array_equal(
            candidate[group].to_numpy()[identity].view(np.uint64),
            baseline[group].to_numpy()[identity].view(np.uint64),
        )


def test_identity_checks_cover_seven_segments() -> None:
    index = pd.date_range("2024-01-01 01:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm")
    baseline = pd.DataFrame(10000.0, index=index, columns=runner.TARGET_COLS)
    checks = runner._identity_checks(baseline, baseline.copy())
    assert [record["segment"] for record in checks] == list(runner.SEGMENTS)
    assert all(record["bit_exact"] for record in checks)


def test_default_namespace_and_config_hash_are_frozen() -> None:
    args = runner.parse_args([])
    assert args.out_dir.as_posix() == "artifacts/postgate/public_fixed_scale095_lowcf025_g1g3_rescue_v1"
    assert hashlib.sha256(Path(args.config).read_bytes()).hexdigest() == runner.CONFIG_SHA256
    assert runner.ACTIVE_GROUPS == ("kpx_group_1", "kpx_group_3")
    assert runner.IDENTITY_GROUP == "kpx_group_2"


def test_existing_output_directory_is_rejected() -> None:
    args = runner.parse_args([])
    with tempfile.TemporaryDirectory() as temporary:
        args.out_dir = Path(temporary)
        with pytest.raises(FileExistsError, match="existing output directory"):
            runner.run(args)

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from scripts import run_public_fixed_scale097_lowcf025 as runner


def _frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01 01:00", periods=5, freq="h")
    return pd.DataFrame(
        {
            "kpx_group_1": np.array([0.0, 0.25, 0.2500000001, 0.9, np.nan]) * 21600.0,
            "kpx_group_2": np.array([0.1, 0.2, 0.25, 0.251, np.nan]) * 21600.0,
            "kpx_group_3": np.array([0.1, 0.249, 0.25, 0.8, np.nan]) * 21000.0,
        },
        index=index,
    )


def test_rule_is_inclusive_at_025_and_uses_exact_097() -> None:
    baseline = _frame()
    candidate, mask = runner.apply_rule(baseline)
    assert mask["kpx_group_1"].tolist() == [True, True, False, False, False]
    assert mask["kpx_group_2"].tolist() == [True, True, True, False, False]
    assert mask["kpx_group_3"].tolist() == [True, True, True, False, False]
    for group in runner.TARGET_COLS:
        np.testing.assert_array_equal(
            candidate.loc[mask[group], group].to_numpy(),
            baseline.loc[mask[group], group].to_numpy() * np.float64(0.97),
        )


def test_identity_rows_are_bit_exact() -> None:
    baseline = _frame()
    candidate, mask = runner.apply_rule(baseline)
    for group in runner.TARGET_COLS:
        identity = ~mask[group].to_numpy()
        np.testing.assert_array_equal(
            candidate[group].to_numpy()[identity].view(np.uint64),
            baseline[group].to_numpy()[identity].view(np.uint64),
        )


def test_stage1_slice_contract_has_17_records_on_synthetic_identity() -> None:
    index = pd.date_range("2023-01-01 01:00", "2024-01-01 00:00", freq="h")
    baseline = pd.DataFrame(12000.0, index=index, columns=runner.TARGET_COLS)
    baseline.loc[index < pd.Timestamp("2023-07-01 01:00"), "kpx_group_3"] = np.nan
    actual = baseline.copy()
    records = runner.parent._stage1_records(actual, baseline, baseline.copy())
    assert len(records) == 17
    assert all(record["delta_total_score"] == 0.0 for record in records)


def test_default_namespace_config_hash_and_constants_are_frozen() -> None:
    args = runner.parse_args([])
    assert args.out_dir.as_posix() == "artifacts/postgate/public_fixed_scale097_lowcf025_v1"
    assert hashlib.sha256(Path(args.config).read_bytes()).hexdigest() == runner.CONFIG_SHA256
    assert runner.FACTOR == np.float64(0.97)
    assert runner.CUTOFF_CF == np.float64(0.25)


def test_existing_output_directory_is_rejected_before_candidate_write() -> None:
    args = runner.parse_args([])
    with tempfile.TemporaryDirectory() as temporary:
        args.out_dir = Path(temporary)
        with pytest.raises(FileExistsError, match="existing output directory"):
            runner.run(args)


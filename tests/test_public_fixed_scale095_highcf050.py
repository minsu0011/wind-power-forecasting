from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from scripts import run_public_fixed_scale095_highcf050 as runner


def _frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01 01:00", periods=6, freq="h")
    return pd.DataFrame(
        {
            "kpx_group_1": np.array([0.0, 0.4999999999, 0.5, 0.8, 1.02, np.nan]) * 21600,
            "kpx_group_2": np.array([0.1, 0.49, 0.5, 0.501, 0.9, np.nan]) * 21600,
            "kpx_group_3": np.array([0.2, 0.499, 0.5, 0.7, 1.0, np.nan]) * 21000,
        },
        index=index,
    )


def test_rule_is_inclusive_at_050_and_uses_exact_095() -> None:
    baseline = _frame()
    candidate, mask = runner._apply_rule(baseline)
    for group in runner.TARGET_COLS:
        assert mask[group].tolist() == [False, False, True, True, True, False]
        np.testing.assert_array_equal(
            candidate.loc[mask[group], group].to_numpy(),
            0.95 * baseline.loc[mask[group], group].to_numpy(),
        )


def test_identity_rows_are_bit_exact_including_nan() -> None:
    baseline = _frame()
    candidate, mask = runner._apply_rule(baseline)
    for group in runner.TARGET_COLS:
        identity = ~mask[group].to_numpy()
        np.testing.assert_array_equal(
            candidate[group].to_numpy()[identity].view(np.uint64),
            baseline[group].to_numpy()[identity].view(np.uint64),
        )


def test_default_namespace_and_preregister_hash_are_frozen() -> None:
    args = runner.parse_args([])
    assert args.out_dir.as_posix() == "artifacts/postgate/public_fixed_scale095_highcf050_v1"
    assert hashlib.sha256(Path(args.config).read_bytes()).hexdigest() == runner.CONFIG_SHA256


def test_existing_output_directory_is_rejected_before_candidate_write() -> None:
    args = runner.parse_args([])
    with tempfile.TemporaryDirectory() as temporary:
        args.out_dir = Path(temporary)
        with pytest.raises(FileExistsError, match="existing output directory"):
            runner.run(args)


def test_stage1_slice_contract_is_17_and_components_are_exact() -> None:
    index = pd.date_range("2023-01-01 01:00", "2024-01-01 00:00", freq="h")
    baseline = pd.DataFrame(12_000.0, index=index, columns=runner.TARGET_COLS)
    baseline.loc[index < pd.Timestamp("2023-07-01 01:00"), "kpx_group_3"] = np.nan
    records = runner.parent._stage1_records(baseline, baseline, baseline.copy())
    checks = runner._stage1_component_checks(records)
    assert len(records) == 17
    assert all(record["delta_total_score"] == 0.0 for record in records)
    assert all(record["passed"] for record in checks.values())


def test_stage1_source_locks_candidate_before_label_reader() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    lock_write = source.index('stage1_lock_path = out_dir / "stage1_candidate_before_label_lock.json"')
    label_read = source.index("labels_2023 = parent._read_stage1_labels")
    assert lock_write < label_read


def test_stage2_source_locks_both_candidates_before_label_reader() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    lock_write = source.index('stage2_lock_path = out_dir / "stage2_candidates_before_label_lock.json"')
    label_read = source.index("labels = _bounded_stage2_labels")
    assert lock_write < label_read


def test_stage1_failure_branch_has_zero_future_access_contract() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert '"2024_baseline_candidate_cells_computed": 0' in source
    assert '"year_2025_baseline_or_sample_value_cells_decoded": 0' in source
    assert "no_rescue" not in source.lower()

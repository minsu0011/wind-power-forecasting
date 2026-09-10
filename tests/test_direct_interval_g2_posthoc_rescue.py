from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_direct_interval_g2_posthoc_rescue as rescue
from scripts import run_direct_interval_selective_transfer as transfer
from scripts import run_ficr_bayes_decision_strict as strict
from src.manifest import sha256_file
from src.metric import CAPACITY_KWH, TARGET_COLS


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_DIR / "configs/direct_interval_g2_posthoc_rescue_preregister_v4.json"


def test_frozen_config_hash_and_risk_disclosure() -> None:
    assert sha256_file(CONFIG) == rescue.CONFIG_SHA
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["risk_classification"]["posthoc_2024_rescue"] is True
    assert payload["risk_classification"]["selection_unsafe"] is True
    assert payload["risk_classification"]["strict_final_isolation"] is False
    assert payload["immutable_rescue_candidate"]["kpx_group_1"] == {"identity": True}
    assert payload["immutable_rescue_candidate"]["kpx_group_3"] == {"identity": True}


def test_frozen_g2_rule_is_single_candidate() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    rule = payload["immutable_rescue_candidate"]
    assert rule["candidate_count"] == 1
    assert rule["kpx_group_2"]["scale_factor"] == 0.98
    assert rule["kpx_group_2"]["utility_advantage_margin"] == 0.01
    assert rule["no_new_factor_margin_lookup_model_surface_threshold_blend_or_group_grid"] is True


def test_identity_check_is_float64_bit_exact() -> None:
    index = pd.date_range("2025-01-01 01:00", periods=2, freq="h")
    baseline = pd.DataFrame(
        {
            "kpx_group_1": [0.0, 1.0],
            "kpx_group_2": [2.0, 3.0],
            "kpx_group_3": [4.0, 5.0],
        },
        index=index,
        dtype=np.float64,
    )
    candidate = baseline.copy()
    candidate["kpx_group_2"] *= 0.98
    rescue.assert_identity_bits(candidate, baseline)
    candidate.loc[index[1], "kpx_group_1"] = np.nextafter(1.0, 2.0)
    with pytest.raises(AssertionError, match="kpx_group_1"):
        rescue.assert_identity_bits(candidate, baseline)


def test_g2_apply_rule_uses_strict_margin_and_exact_factor() -> None:
    capacity = CAPACITY_KWH[rescue.GROUP]
    baseline_cf = np.array([0.50, 0.50], dtype=np.float64)
    baseline = pd.Series(baseline_cf * capacity)
    utility = np.zeros((2, 103), dtype=np.float64)
    # U(.49)-U(.50): first is exactly .01 (must not gate), second is .02.
    utility[0, 49], utility[0, 50] = 0.01, 0.0
    utility[1, 49], utility[1, 50] = 0.02, 0.0
    selected, detail = transfer._apply_spec(
        baseline, utility, group=rescue.GROUP, spec=rescue.SPEC
    )
    assert detail["gate"].tolist() == [False, True]
    assert selected.iloc[0] == baseline.iloc[0]
    assert selected.iloc[1] == 0.98 * baseline.iloc[1]


def test_csv_validator_enforces_bom_schema_time_and_six_decimals(tmp_path: Path) -> None:
    index = pd.date_range("2025-01-01 01:00", periods=8760, freq="h", name="forecast_kst_dtm")
    sample = pd.DataFrame(
        {
            "forecast_id": pd.Series([f"id-{i}" for i in range(8760)], dtype="string"),
            "forecast_kst_dtm": pd.Series(index.astype(str), dtype="string"),
            **{group: np.zeros(8760) for group in TARGET_COLS},
        }
    )
    candidate = pd.DataFrame(
        {
            group: np.linspace(0.0, 0.5 * CAPACITY_KWH[group], 8760)
            for group in TARGET_COLS
        }
    )
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = candidate[group]
    path = tmp_path / "candidate.csv"
    strict._atomic_csv(submission, path)
    result = rescue._validate_csv(path, sample, candidate)
    assert result["utf8_bom"] is True
    assert result["rows"] == 8760
    assert result["six_decimal_text_roundtrip"] is True


def test_default_paths_are_new_rescue_namespace() -> None:
    args = rescue.parse_args(["--stage", "prescore"])
    assert args.config.as_posix().endswith("direct_interval_g2_posthoc_rescue_preregister_v4.json")
    assert args.out_dir.as_posix().endswith("direct_interval_g2_posthoc_rescue_v4")


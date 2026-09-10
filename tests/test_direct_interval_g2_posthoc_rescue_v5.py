from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_direct_interval_g2_posthoc_rescue_v5 as rescue
from src.manifest import sha256_file
from src.metric import CAPACITY_KWH


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_DIR / "configs/direct_interval_g2_posthoc_rescue_preregister_v5.json"


def test_config_hash_risk_and_immutable_candidate() -> None:
    assert sha256_file(CONFIG) == rescue.CONFIG_SHA
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["risk_classification"] == {
        "posthoc_2023_selected": True,
        "posthoc_2024_rescue": True,
        "selection_unsafe": True,
        "strict_final_isolation": False,
        "private_champion": False,
        "leaderboard_score_claim": False,
    }
    assert payload["immutable_candidate"]["kpx_group_2"]["factor"] == 0.98
    assert payload["immutable_candidate"]["kpx_group_2"]["margin"] == 0.01
    assert payload["immutable_candidate"]["no_factor_margin_lookup_model_feature_parameter_seed_data_group_or_threshold_change_from_v4"] is True


def test_recursive_ast_local_import_closure_is_exact_and_minimal() -> None:
    paths = rescue.resolve_ast_local_import_closure(
        PROJECT_DIR / "scripts/run_direct_interval_g2_posthoc_rescue_v5.py"
    )
    relative = [path.relative_to(PROJECT_DIR).as_posix() for path in paths]
    assert relative == [
        "scripts/run_direct_interval_g2_posthoc_rescue_v5.py",
        "src/__init__.py",
        "src/direct_interval_probability.py",
        "src/manifest.py",
        "src/metric.py",
    ]
    assert "scripts/run_direct_interval_selective_transfer.py" not in relative
    assert "scripts/run_catboost_multiquantile_bayes.py" not in relative


def test_ast_resolver_rejects_unresolved_local_import(tmp_path: Path) -> None:
    # The resolver only accepts entrypoints under the project; this also prevents
    # a test fixture from silently broadening its authority outside the repo.
    fixture = tmp_path / "bad.py"
    fixture.write_text("from src.missing_local_module import value\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="escapes project"):
        rescue.resolve_ast_local_import_closure(fixture)


def test_apply_rule_uses_linear_lookup_strict_margin_and_exact_factor() -> None:
    capacity = CAPACITY_KWH[rescue.GROUP]
    baseline = pd.Series(np.array([0.5, 0.5]) * capacity)
    utility = np.zeros((2, 103), dtype=np.float64)
    utility[0, 49], utility[0, 50] = 0.01, 0.0
    utility[1, 49], utility[1, 50] = 0.02, 0.0
    candidate, detail = rescue.apply_rule(baseline, utility)
    assert detail["gate"].tolist() == [False, True]
    assert candidate.iloc[0] == baseline.iloc[0]
    assert candidate.iloc[1] == 0.98 * baseline.iloc[1]


def test_identity_assertion_is_float64_bit_exact() -> None:
    index = pd.date_range("2025-01-01 01:00", periods=2, freq="h")
    baseline = pd.DataFrame(
        {
            "kpx_group_1": [1.0, 2.0],
            "kpx_group_2": [3.0, 4.0],
            "kpx_group_3": [5.0, 6.0],
        }, index=index, dtype=np.float64,
    )
    candidate = baseline.copy()
    candidate["kpx_group_2"] *= 0.98
    rescue._assert_identity(candidate, baseline)
    candidate.iloc[0, 2] = np.nextafter(candidate.iloc[0, 2], np.inf)
    with pytest.raises(AssertionError, match="kpx_group_3"):
        rescue._assert_identity(candidate, baseline)


def test_v4_is_quarantined_and_submission_unauthorized() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = PROJECT_DIR / payload["supersession"]["quarantined_v4_root"]
    assert root.is_dir()
    assert payload["supersession"]["v4_submission_authorized"] is False
    assert not (PROJECT_DIR / "artifacts/postgate/direct_interval_g2_posthoc_rescue_v4").exists()


from __future__ import annotations

from decimal import Decimal
import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from scripts import build_public_group_scale095_probe as builder
from src.metric import TARGET_COLS
from src.public_group_scale_probe import macro_separability_residuals, scale_one_group


def test_config_hash_gate_and_no_recommendation_are_frozen() -> None:
    args = builder.parse_args([])
    assert args.out_dir.as_posix() == "artifacts/postgate/public_group_scale095_probe"
    assert hashlib.sha256(Path(args.config).read_bytes()).hexdigest() == builder.CONFIG_SHA256
    config = builder._verify_config(Path(args.config))
    gate = config["activation_gate"]
    advantage = Decimal(str(gate["global_scale_095"]["score"])) - Decimal(str(gate["base"]["score"]))
    assert advantage == Decimal("0.0006668153")
    assert gate["passed"] is True
    assert config["submission_contract"]["automatic_final_recommendation"] is False
    assert config["submission_contract"]["included_in_final_top2_selection"] is False


def test_factor095_scales_only_selected_group_bit_exactly() -> None:
    index = pd.date_range("2025-01-01 01:00", periods=4, freq="h", name="forecast_kst_dtm")
    base = pd.DataFrame(
        {
            "kpx_group_1": [1000.0, 2000.0, 3000.0, 4000.0],
            "kpx_group_2": [5000.0, 6000.0, 7000.0, 8000.0],
            "kpx_group_3": [9000.0, 10000.0, 11000.0, 12000.0],
        },
        index=index,
        dtype=np.float64,
    )
    candidate = scale_one_group(base, "kpx_group_2", factor=0.95)
    np.testing.assert_array_equal(candidate["kpx_group_2"], base["kpx_group_2"] * 0.95)
    for group in ("kpx_group_1", "kpx_group_3"):
        np.testing.assert_array_equal(
            candidate[group].to_numpy().view(np.uint64),
            base[group].to_numpy().view(np.uint64),
        )


def test_macro_additivity_contract_is_componentwise_exact() -> None:
    base = {"score": 0.6, "one_minus_nmae": 0.8, "ficr": 0.4}
    group_only = {
        "kpx_group_1": {"score": 0.61, "one_minus_nmae": 0.82, "ficr": 0.40},
        "kpx_group_2": {"score": 0.59, "one_minus_nmae": 0.79, "ficr": 0.39},
        "kpx_group_3": {"score": 0.605, "one_minus_nmae": 0.80, "ficr": 0.41},
    }
    global_scaled = {
        component: base[component] + sum(group_only[group][component] - base[component] for group in TARGET_COLS)
        for component in ("score", "one_minus_nmae", "ficr")
    }
    residuals = macro_separability_residuals(base=base, global_scaled=global_scaled, group_only=group_only)
    assert all(abs(item["residual"]) < 1e-15 for item in residuals.values())


def test_recursive_source_closure_is_complete_for_local_imports() -> None:
    paths = {
        path.relative_to(builder.PROJECT_ROOT).as_posix()
        for path in builder.resolve_ast_closure(Path(builder.__file__))
    }
    assert "scripts/build_public_group_scale095_probe.py" in paths
    assert "src/public_group_scale_probe.py" in paths
    assert "src/public_scale_probe.py" in paths
    assert "src/metric.py" in paths
    assert "src/manifest.py" in paths


def test_existing_output_directory_is_rejected_postrun_safely() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        with pytest.raises(FileExistsError, match="existing output directory"):
            builder.preflight(Path(temporary))


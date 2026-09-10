from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_copernicus_dem_directional_exposure as runner
from src.copernicus_dem_exposure import (
    EXTENDED_COLUMNS,
    TERRAIN_VALUE_COLUMNS,
    WIND_COLUMNS,
    build_dynamic_exposure_features,
    parse_turbines,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/copernicus_dem_directional_exposure_paired_increment_preregister_v1.json"
CONFIG_SHA = "21e2a6aaee02d11c00d2b8a266e30917efa4f5a505417ceadb9f618fe7d2443b"
TABLE = ROOT / "artifacts/external/copernicus_dem_glo30_2021_terrain_v1/features_v1/static_directional_terrain_17x16.parquet"
TABLE_SHA = "71945cab5a07c328eabc5a8fd01642eaffb68194e23efe921f1ba53317482fae"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict:
    assert _sha(CONFIG) == CONFIG_SHA
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _synthetic_table(config: dict) -> pd.DataFrame:
    rows = []
    for turbine in config["turbines"]["rows"]:
        for sector in range(16):
            record = {
                "turbine_id": turbine["id"],
                "group": turbine["group"],
                "capacity_mw": turbine["capacity_mw"],
                "sector_index": sector,
            }
            for offset, name in enumerate(TERRAIN_VALUE_COLUMNS):
                record[name] = 1000.0 * turbine["id"] + 10.0 * sector + offset
            rows.append(record)
    return pd.DataFrame(rows)


def test_preregister_is_single_fixed_candidate() -> None:
    config = _config()
    assert config["risk_and_scope"]["candidate_count"] == 1
    assert config["paired_model"]["transfer_weight"] == 0.05
    assert config["paired_model"]["parameters"]["random_state"] == 42
    assert config["paired_model"]["parameters"]["n_estimators"] == 1500
    assert config["dynamic_feature_algorithm"]["extended_features_after_612_in_exact_order"] == list(EXTENDED_COLUMNS)
    assert config["static_terrain_algorithm"]["horizon_radii_m"] == [500.0, 1000.0, 2000.0, 5000.0]
    assert len(config["static_terrain_algorithm"]["sector_centres_degrees_clockwise_from_north"]) == 16


def test_static_table_identity_shape_and_finite() -> None:
    assert _sha(TABLE) == TABLE_SHA
    table = pd.read_parquet(TABLE)
    assert len(table) == 17 * 16
    assert table["turbine_id"].nunique() == 17
    assert table["sector_index"].nunique() == 16
    assert table.duplicated(["turbine_id", "sector_index"]).sum() == 0
    assert np.isfinite(table[list(TERRAIN_VALUE_COLUMNS)].to_numpy(np.float64)).all()


@pytest.mark.parametrize(
    ("u", "v", "expected_sector"),
    [
        (0.0, -1.0, 0),
        (-1.0, 0.0, 4),
        (0.0, 1.0, 8),
        (1.0, 0.0, 12),
    ],
)
def test_meteorological_source_bearing_sector(u: float, v: float, expected_sector: int) -> None:
    config = _config()
    turbines = parse_turbines(config["turbines"]["rows"])
    control = pd.DataFrame(
        [[u, u, v, v]],
        columns=WIND_COLUMNS,
        index=pd.DatetimeIndex(["2023-01-01 01:00"], name="forecast_kst_dtm"),
    )
    result = build_dynamic_exposure_features(control, _synthetic_table(config), turbines, group="kpx_group_1")
    expected_mean = np.mean([1000.0 * i + 10.0 * expected_sector for i in range(1, 7)])
    assert result.iloc[0, 0] == pytest.approx(expected_mean)
    assert tuple(result.columns) == EXTENDED_COLUMNS
    assert all(dtype == np.dtype("float32") for dtype in result.dtypes)


def test_capacity_weighted_population_mean_and_std() -> None:
    config = _config()
    turbines = parse_turbines(config["turbines"]["rows"])
    control = pd.DataFrame(
        [[0.0, 0.0, -1.0, -1.0]], columns=WIND_COLUMNS, index=pd.DatetimeIndex(["2023-01-01 01:00"])
    )
    result = build_dynamic_exposure_features(control, _synthetic_table(config), turbines, group="kpx_group_3")
    values = np.asarray([13000.0, 14000.0, 15000.0, 16000.0, 17000.0])
    assert result.iloc[0]["copdem__signed_slope_alignment_cw_mean"] == pytest.approx(values.mean())
    assert result.iloc[0]["copdem__signed_slope_alignment_cw_std"] == pytest.approx(values.std(ddof=0))


def test_nonfinite_direction_fails_closed() -> None:
    config = _config()
    turbines = parse_turbines(config["turbines"]["rows"])
    control = pd.DataFrame([[np.nan, 0.0, 0.0, 0.0]], columns=WIND_COLUMNS)
    with pytest.raises(ValueError, match="nonfinite direction"):
        build_dynamic_exposure_features(control, _synthetic_table(config), turbines, group="kpx_group_1")


def test_feature_module_has_no_label_metric_or_network_reader() -> None:
    source = (ROOT / "src/copernicus_dem_exposure.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "requests" not in imports
    assert "urllib" not in imports
    assert "src.metric" not in source
    assert "read_csv" not in source
    assert "train_labels" not in source


def test_runner_verifies_candidate_lock_before_label_source_resolution(tmp_path: Path) -> None:
    config = _config()
    config["official_data_and_lineage"]["label_file"] = {
        "path": str(tmp_path / "must_not_be_opened.csv"),
        "bytes": 1,
        "sha256": "0" * 64,
    }
    with pytest.raises(FileNotFoundError, match="missing-lock"):
        runner.read_score_labels(
            runner.parse_args([]),
            config,
            candidate_lock={"path": str(tmp_path / "missing-lock.json"), "size_bytes": 1, "sha256": "0" * 64},
        )


def test_runner_contract_literals_and_guard_identity() -> None:
    config = _config()
    assert runner.CONFIG_SHA == CONFIG_SHA
    assert runner.STATIC_TABLE_SHA == TABLE_SHA
    assert runner.EXPERIMENT_ID == config["experiment_id"]
    assert runner.TRANSFER_WEIGHT == 0.05
    assert runner.LABEL_SLICES["g12_fit"]["end"] == 340783
    assert runner.LABEL_SLICES["g12_score"]["start"] == 340783
    assert runner.LABEL_SLICES["g3_fit"]["end"] == runner.LABEL_SLICES["g3_score"]["start"]
    source = (ROOT / "scripts/run_copernicus_dem_directional_exposure.py").read_text(encoding="utf-8")
    assert '"experiment_id": EXPERIMENT_ID' in source
    assert "stage1_candidate_before_score_labels_lock.json" in source
    assert source.index("candidate_lock = describe_file(lock_path)") < source.index("score_labels, score_evidence = read_score_labels")
    assert "2025_contest_values_read\": 0" in source


def test_runner_paired_model_parameters_match_preregister() -> None:
    from src.jma_paired_increment import MODEL_PARAMETERS

    config = _config()
    assert MODEL_PARAMETERS == config["paired_model"]["parameters"]

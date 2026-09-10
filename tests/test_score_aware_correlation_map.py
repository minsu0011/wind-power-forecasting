from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.score_aware_correlation_map import (
    TARGET_ORDER,
    CorrelationMapConfig,
    build_correlation_map,
    write_correlation_map,
)


def _synthetic_oof() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_per_slice = 24
    slice_names = (
        ("G1", "2023_h1"),
        ("G1", "2023_h2"),
        ("G2", "2023_h1"),
        ("G2", "2023_h2"),
    )
    x = np.tile(np.linspace(0.05, 0.95, rows_per_slice), len(slice_names))
    rng = np.random.default_rng(20260810)
    features = pd.DataFrame(
        {
            "gfs__idw__ws100": x,
            "gfs__nearest__ws100": 7.0 + 2.0 * x,
            "ldaps__idw__heightAboveGround_2_2t": rng.normal(size=len(x)),
            "time__hour_sin": np.tile(
                np.sin(np.arange(rows_per_slice) * 2.0 * np.pi / 24.0),
                len(slice_names),
            ),
            "time__lead_hours": np.tile(
                np.arange(1, rows_per_slice + 1), len(slice_names)
            ),
        },
        index=pd.Index(np.arange(1000, 1000 + len(x)), name="row_id"),
    )
    groups = np.concatenate(
        [np.repeat(group, rows_per_slice) for group, _ in slice_names]
    )
    folds = np.concatenate(
        [np.repeat(fold, rows_per_slice) for _, fold in slice_names]
    )
    actual = 50.0 + 2.0 * np.sin(np.arange(len(x)) * 0.3)
    for start in range(0, len(x), rows_per_slice):
        actual[start : start + 2] = 5.0
    baseline = actual - 10.0 * x
    forecast_times = np.concatenate(
        [
            pd.date_range(start, periods=rows_per_slice, freq="h").to_numpy()
            for start in (
                "2022-01-01",
                "2022-04-01",
                "2022-07-01",
                "2022-10-01",
            )
        ]
    )
    rows = pd.DataFrame(
        {
            "group": groups,
            "fold": folds,
            "actual_kwh": actual,
            "baseline_kwh": baseline,
            "capacity_kwh": 100.0,
            "forecast_kst_dtm": forecast_times,
        },
        index=features.index,
    )
    return features, rows


def _config() -> CorrelationMapConfig:
    return CorrelationMapConfig(
        min_pair_rows=8,
        min_slice_rows=8,
        max_redundancy_rows_per_source=100,
        n_jobs=2,
    )


def test_score_aware_map_recovers_residual_boundary_and_stability() -> None:
    features, rows = _synthetic_oof()
    result = build_correlation_map(
        oof_features=features,
        oof_rows=rows,
        redundancy_sources={
            "causal_train": features.iloc[::2],
            "causal_apply": features.iloc[1::2],
        },
        config=_config(),
    )

    assert len(result.feature_map) == len(features.columns) * len(TARGET_ORDER)
    residual = result.feature_map.query(
        "feature == 'gfs__idw__ws100' and target == 'signed_residual_cf'"
    ).iloc[0]
    assert residual["spearman"] > 0.999
    assert residual["weighted_spearman"] > 0.999
    assert residual["rank_normalized_pearson"] > 0.999
    assert residual["spearman_sign_slice_count"] >= 4
    assert residual["spearman_sign_stability"] == 1.0
    assert residual["group_sign_stability"] == 1.0
    assert residual["fold_sign_stability"] == 1.0

    within_six = result.feature_map.query(
        "feature == 'gfs__idw__ws100' and target == 'within_6'"
    ).iloc[0]
    assert within_six["energy_weighted_auc"] < 0.01
    assert within_six["weighted_auc_signed_edge"] < -0.98
    assert within_six["rank_biserial"] == within_six["weighted_auc_signed_edge"]
    assert within_six["weighted_auc_skill"] > 0.98
    assert within_six["auc_sign_slice_count"] >= 4
    assert within_six["auc_sign_stability"] == 1.0

    utility = result.feature_map.query(
        "feature == 'gfs__idw__ws100' and target_kind == 'exact_utility'"
    )
    assert set(utility["target"]) == {
        "plus_vs_minus_utility",
        "best_one_percent_utility_gain",
    }
    assert utility["spearman"].notna().any()

    wind_family = result.family_map.query(
        "physical_family == 'wind' and target == 'signed_residual_cf'"
    ).iloc[0]
    assert wind_family["feature_count"] == 2
    assert wind_family["top_feature"] in {
        "gfs__idw__ws100",
        "gfs__nearest__ws100",
    }
    assert set(result.metadata["registered_strata"]) == {
        "group",
        "fold",
        "lead_bin",
        "season",
        "causal_control_cf_bin",
    }


def test_redundancy_cluster_uses_absolute_pooled_spearman() -> None:
    features, rows = _synthetic_oof()
    # An inverted copy is redundant too because the frozen criterion is |rho|.
    train = features.copy()
    apply = features.copy()
    train["gfs__nearest__ws100"] = -train["gfs__idw__ws100"]
    apply["gfs__nearest__ws100"] = -apply["gfs__idw__ws100"]
    result = build_correlation_map(
        oof_features=features,
        oof_rows=rows,
        redundancy_sources={"train": train, "apply": apply},
        config=_config(),
    )
    members = result.redundancy_membership.set_index("feature")
    first = members.loc["gfs__idw__ws100"]
    second = members.loc["gfs__nearest__ws100"]
    assert first["redundancy_cluster_id"] == second["redundancy_cluster_id"]
    assert first["redundancy_cluster_size"] >= 2
    edge = result.redundancy_edges.query(
        "feature_a == 'gfs__idw__ws100' and "
        "feature_b == 'gfs__nearest__ws100'"
    ).iloc[0]
    assert edge["pooled_spearman"] < -0.999
    assert edge["pooled_abs_spearman"] > 0.999


def test_writer_emits_parquet_json_hash_manifest(tmp_path) -> None:
    features, rows = _synthetic_oof()
    result = build_correlation_map(
        oof_features=features,
        oof_rows=rows,
        config=_config(),
    )
    paths = write_correlation_map(result, tmp_path / "map")
    assert set(paths) == {
        "feature_map_parquet",
        "feature_map_json",
        "family_map_parquet",
        "family_map_json",
        "redundancy_membership_parquet",
        "redundancy_membership_json",
        "redundancy_edges_parquet",
        "redundancy_edges_json",
        "metadata_json",
    }
    assert all(path.exists() for path in paths.values())
    written = pd.read_parquet(paths["feature_map_parquet"])
    assert len(written) == len(result.feature_map)
    with paths["metadata_json"].open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["schema_version"] == "score_aware_correlation_map_v1"
    assert "not a Public or Private" in metadata["interpretation"]
    assert len(metadata["output_sha256"]) == 8
    assert all(len(value) == 64 for value in metadata["output_sha256"].values())
    with pytest.raises(FileExistsError, match="no-overwrite"):
        write_correlation_map(result, tmp_path / "map")


def test_rejects_index_misalignment_and_more_than_seven_jobs() -> None:
    features, rows = _synthetic_oof()
    shifted = rows.copy()
    shifted.index = shifted.index + 1
    with pytest.raises(ValueError, match="indices must be exactly aligned"):
        build_correlation_map(
            oof_features=features,
            oof_rows=shifted,
            config=_config(),
        )
    with pytest.raises(ValueError, match="between 1 and 7"):
        CorrelationMapConfig(n_jobs=8)

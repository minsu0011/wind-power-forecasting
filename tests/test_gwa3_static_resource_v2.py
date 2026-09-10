from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_gwa3_static_resource_stage1 as quarantined_v1
from scripts import run_gwa3_static_resource_v2 as runner
from src.gwa3_static_resource import EXTENDED_COLUMNS, build_gwa3_features


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_remains_permanently_quarantined() -> None:
    with pytest.raises(RuntimeError, match="PERMANENTLY QUARANTINED"):
        quarantined_v1.main()


def test_v2_preregister_and_frozen_resources_are_exact() -> None:
    assert _sha(ROOT / "configs/gwa3_static_resource_paired_increment_preregister_v2.json") == runner.CONFIG_SHA
    assert _sha(ROOT / "src/gwa3_static_resource.py") == runner.FEATURE_SHA
    assert _sha(runner.STATIC_PATH) == runner.STATIC_SHA
    assert _sha(runner.SOURCE_MANIFEST_PATH) == runner.MANIFEST_SHA
    config = json.loads(
        (ROOT / "configs/gwa3_static_resource_paired_increment_preregister_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["experiment_id"] == runner.EXPERIMENT_ID
    assert config["v1_quarantine"]["status"] == "PERMANENTLY_QUARANTINED_AND_NOT_EVIDENCE_FOR_V2"
    assert config["single_candidate"]["candidate_count"] == 1
    assert config["single_candidate"]["transfer_weight"] == 0.1


def test_v2_has_local_guard_and_no_worldcover_guard_import() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "run_esa_worldcover2020_directional_landcover" not in source
    assert "world.HeavyFitGuard" not in source
    assert 'EXPERIMENT_ID = "gwa3_static_resource_paired_increment_strict_v2"' in source
    assert "class HeavyFitGuard" in source


def test_guard_exact_identity_and_wrong_token_release_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "guard.json"
    guard = runner.HeavyFitGuard(path)
    guard.__enter__()
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["experiment_id"] == runner.EXPERIMENT_ID
        assert record["pid"] == __import__("os").getpid()
        corrupted = {**record, "token": "wrong-token"}
        path.write_text(json.dumps(corrupted), encoding="utf-8")
        with pytest.raises(AssertionError, match="ownership/identity changed"):
            guard.release()
        assert path.exists()
        path.write_text(json.dumps(record), encoding="utf-8")
        guard.verify_live()
    finally:
        if guard.acquired:
            guard.release()
    assert not path.exists()


def test_candidate_locks_precede_score_readers_in_each_stage() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    stage1 = source[source.index("def _run_stage1(") : source.index("def _load_stage2_baseline(")]
    assert stage1.index('lock_path = args.out_dir / "stage1_candidate_before_score_labels_lock.json"') < stage1.index(
        "score_labels, score_evidence = _stage1_score_labels"
    )
    stage2 = source[source.index("def _run_stage2(") : source.index("def _run_final(")]
    assert stage2.index('lock_path = args.out_dir / "stage2_candidate_before_2024_score_labels_lock.json"') < stage2.index(
        "score_labels, score_evidence = _bounded_label_slice"
    )
    pipeline = source[source.index("def run(args:") : source.index("def main(")]
    assert pipeline.index("if not stage1_pass:") < pipeline.index("_run_stage2(")
    assert pipeline.index("if not stage2_pass:") < pipeline.index("_run_final(")


def test_gwa_feature_contract_is_exact_and_finite() -> None:
    static = pd.read_parquet(runner.STATIC_PATH)
    index = pd.date_range("2023-01-01 01:00", periods=5, freq="h", name="forecast_kst_dtm")
    control = pd.DataFrame(
        {
            "ldaps__idw__heightAboveGround_50_50MUmax": [2, 3, 4, 5, 6],
            "ldaps__idw__heightAboveGround_50_50MUmin": [1, 2, 3, 4, 5],
            "ldaps__idw__heightAboveGround_50_50MVmax": [4, 3, 2, 1, 0],
            "ldaps__idw__heightAboveGround_50_50MVmin": [3, 2, 1, 0, -1],
            "gfs__idw__heightAboveGround_100_100u": [1, 2, 3, 4, 5],
            "gfs__idw__heightAboveGround_100_100v": [5, 4, 3, 2, 1],
            "ldaps__idw__hub_ws": [4, 5, 6, 7, 8],
            "gfs__idw__hub_ws": [5, 6, 7, 8, 9],
        },
        index=index,
        dtype=np.float32,
    )
    result = build_gwa3_features(control, static, group="kpx_group_1")
    assert result.index.equals(index)
    assert tuple(result.columns) == EXTENDED_COLUMNS
    assert result.shape == (5, 24)
    assert np.isfinite(result.to_numpy()).all()


def test_strict_gate_definitions_are_not_rescue_or_group_selective() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "alternate transfer" not in source.lower()
    assert "public subgroup" not in source.lower()
    assert "TRANSFER_WEIGHT = 0.10" in source
    assert "all(value > 0.0 for value in full_deltas)" in source
    assert "positive_segments >= 5" in source
    assert "all(value > 0.0 for value in group_full)" in source

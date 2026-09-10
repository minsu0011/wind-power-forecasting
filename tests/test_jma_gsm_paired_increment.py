from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_jma_gsm_paired_increment as runner
from src.jma_paired_increment import (
    CANDIDATES,
    JMA_COLUMNS,
    MODEL_PARAMETERS,
    TRANSFER_WEIGHT,
    build_jma_features,
    extended_features,
    paired_increment,
    select_stage1_candidate,
    transfer_increment,
)


ROOT = Path(__file__).resolve().parents[1]
V1_SHA = "b3f03c8d022ee212bdbbdfc9f707b6f54b6b8965ad9c338bf15985a5e1597aa2"
V2_SHA = "99356308ebddfd5acd4b8d6c700c04e701a94c59bc32eb788ab5c5a65e46909e"
V3_SHA = "cb5ff8ebb613230e67487499acfcc8f4030d55d637d43d357582d76a72b7103e"
V4_SHA = "f1568dd08d1b9ad8e6d5667486e90b2608b994954aa6dd3cb2aa70864d640f80"
V5_SHA = "c0a3dc25366b518fcc3fdf3ece4e7105ddefe3a3bfe7bb97b717da4838a2791a"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source() -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.Series]:
    index = pd.date_range("2023-01-01 00:00", periods=24, freq="h", name="forecast_kst_dtm")
    frame = pd.DataFrame(
        {
            "group": "kpx_group_1",
            "time": index,
            "wind_speed_10m_previous_day1": np.arange(24, dtype=float) + 100.0,
            "wind_direction_10m_previous_day1": 90.0,
            "wind_speed_10m_previous_day2": np.arange(24, dtype=float) + 10.0,
            "wind_direction_10m_previous_day2": 180.0,
        }
    )
    cross = pd.Series(5.0, index=index)
    return frame, index, cross


def test_preregistration_hashes_and_protocol_supersession() -> None:
    v1 = ROOT / "configs/jma_gsm_paired_increment_preregister_v1.json"
    v2 = ROOT / "configs/jma_gsm_paired_increment_preregister_v2.json"
    v3 = ROOT / "configs/jma_gsm_paired_increment_preregister_v3.json"
    v4 = ROOT / "configs/jma_gsm_paired_increment_preregister_v4.json"
    v5 = ROOT / "configs/jma_gsm_paired_increment_preregister_v5.json"
    assert _sha(v1) == V1_SHA
    assert _sha(v2) == V2_SHA
    assert _sha(v3) == V3_SHA
    assert _sha(v4) == V4_SHA
    assert _sha(v5) == V5_SHA
    payload = json.loads(v2.read_text(encoding="utf-8"))
    assert payload["base_preregister"]["sha256"] == V1_SHA
    assert payload["supersession"]["v1_candidate_models_fit"] == 0
    assert payload["effective_contract"]["boundary_row"]["paired_increment_cf"] == 0.0
    assert payload["effective_contract"]["prohibitions"][
        "request_or_parse_any_2025_JMA_value_before_full_Stage2_pass_and_separate_final_prescore_lock"
    ]
    v3_payload = json.loads(v3.read_text(encoding="utf-8"))
    assert v3_payload["base_preregister_v2"]["sha256"] == V2_SHA
    assert v3_payload["v2_protocol_failure"]["candidate_metric_values_computed"] == 0
    assert v3_payload["immutable_inheritance"]["new_candidates_or_weights"] == 0
    assert v3_payload["lock_order"]["score_reader_fail_closed_without_exact_lock"]
    v4_payload = json.loads(v4.read_text(encoding="utf-8"))
    assert v4_payload["base_v3_preregister"]["sha256"] == V3_SHA
    assert v4_payload["supersession"]["v3_candidate_models_fit"] == 0
    assert v4_payload["complete_v2_incident_ledger"]["candidate_metric_values_computed"] == 0
    assert v4_payload["stage1_boundary_identity_patch"]["paired_increment_cf"] == 0.0
    v5_payload = json.loads(v5.read_text(encoding="utf-8"))
    assert v5_payload["base_v4_preregister"]["sha256"] == V4_SHA
    assert v5_payload["v4_failure_and_quarantine"]["candidate_metric_values_computed"] == 0
    assert v5_payload["immutable_inheritance"]["new_candidate_or_weight_count"] == 0
    assert v5_payload["effective_score_reader_patch"]["metric_function_unchanged"]


def test_candidate_A_uses_day2_at_every_hour() -> None:
    source, index, cross = _source()
    result = build_jma_features(
        source,
        group="kpx_group_1",
        index=index,
        candidate=CANDIDATES[0],
        cross_hub_ws_mean=cross,
    )
    np.testing.assert_allclose(result[JMA_COLUMNS[0]], np.arange(24) + 10.0)
    np.testing.assert_allclose(result[JMA_COLUMNS[1]], 0.0, atol=2e-6)
    np.testing.assert_allclose(result[JMA_COLUMNS[2]], np.arange(24) + 10.0, atol=2e-6)
    np.testing.assert_allclose(result[JMA_COLUMNS[3]], np.arange(24) + 5.0)


def test_candidate_B_uses_day1_only_hours_01_through_13() -> None:
    source, index, cross = _source()
    result = build_jma_features(
        source,
        group="kpx_group_1",
        index=index,
        candidate=CANDIDATES[1],
        cross_hub_ws_mean=cross,
    )
    expected = np.arange(24, dtype=float) + 10.0
    expected[1:14] = np.arange(1, 14, dtype=float) + 100.0
    np.testing.assert_array_equal(result[JMA_COLUMNS[0]].to_numpy(), expected.astype(np.float32))
    assert result.iloc[0, 0] == 10.0
    assert result.iloc[14, 0] == 24.0


def test_extended_schema_and_increment_transfer_are_exact() -> None:
    source, index, cross = _source()
    control = pd.DataFrame(
        {"a": np.arange(24), "cross__hub_ws_mean": cross}, index=index, dtype=np.float32
    )
    jma = build_jma_features(
        source,
        group="kpx_group_1",
        index=index,
        candidate=CANDIDATES[0],
        cross_hub_ws_mean=cross,
    )
    extended = extended_features(control, jma)
    assert tuple(extended.columns) == ("a", "cross__hub_ws_mean", *JMA_COLUMNS)
    increment = paired_increment(np.array([-1.0, 0.5, 2.0]), np.array([0.2, 0.4, 0.8]))
    np.testing.assert_allclose(increment, np.array([0.2, -0.1, -0.22]), rtol=0.0, atol=3e-17)
    baseline = pd.Series([1000.0, 21500.0, 100.0], index=index[:3], name="kpx_group_1")
    candidate = transfer_increment(baseline, increment, capacity_kwh=21600.0)
    expected = np.clip(baseline.to_numpy() + 0.25 * 21600.0 * increment, 0.0, 1.02 * 21600.0)
    np.testing.assert_array_equal(candidate.to_numpy(), expected)


def test_stage1_selection_is_all_17_positive_maximin_then_mean_then_A() -> None:
    fail, audit = select_stage1_candidate(
        {CANDIDATES[0]: np.ones(17), CANDIDATES[1]: np.r_[np.ones(16), 0.0]}
    )
    assert fail == CANDIDATES[0]
    assert not audit["candidates"][CANDIDATES[1]]["passed"]
    winner, _ = select_stage1_candidate(
        {CANDIDATES[0]: np.full(17, 0.1), CANDIDATES[1]: np.r_[np.full(16, 0.2), 0.11]}
    )
    assert winner == CANDIDATES[1]
    tie, _ = select_stage1_candidate(
        {CANDIDATES[0]: np.full(17, 0.1), CANDIDATES[1]: np.full(17, 0.1)}
    )
    assert tie == CANDIDATES[0]
    none, _ = select_stage1_candidate(
        {CANDIDATES[0]: np.r_[np.ones(16), -1e-12], CANDIDATES[1]: np.zeros(17)}
    )
    assert none is None


def test_immutable_model_contract() -> None:
    assert CANDIDATES == (
        "A_day2_w025",
        "B_day1_hours01_13_else_day2_w025",
    )
    assert TRANSFER_WEIGHT == 0.25
    assert MODEL_PARAMETERS == {
        "objective": "regression_l1",
        "n_estimators": 1500,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "random_state": 42,
        "n_jobs": 7,
    }


def test_missing_or_misaligned_jma_is_rejected() -> None:
    source, index, cross = _source()
    with pytest.raises(ValueError, match="coverage"):
        build_jma_features(
            source.iloc[:-1],
            group="kpx_group_1",
            index=index,
            candidate=CANDIDATES[0],
            cross_hub_ws_mean=cross,
        )


def test_runner_config_and_ast_closure_are_frozen() -> None:
    v1, v2, v3, v4, v5 = runner.verify_config(
        ROOT / "configs/jma_gsm_paired_increment_preregister_v5.json"
    )
    assert v1["experiment_id"] == "jma_gsm_paired_increment_strict_v1"
    assert v2["experiment_id"] == "jma_gsm_paired_increment_strict_v2"
    assert v3["experiment_id"] == "jma_gsm_paired_increment_strict_v3"
    assert v4["experiment_id"] == "jma_gsm_paired_increment_strict_v4"
    assert v5["experiment_id"] == "jma_gsm_paired_increment_strict_v5"
    closure = runner.resolve_ast_local_import_closure(
        ROOT / "scripts/run_jma_gsm_paired_increment.py"
    )
    relative = {path.relative_to(ROOT).as_posix() for path in closure}
    assert "scripts/run_jma_gsm_paired_increment.py" in relative
    assert "scripts/run_shared_q07_multiseed.py" in relative
    assert "src/jma_paired_increment.py" in relative
    assert "src/features.py" in relative
    assert "src/metric.py" in relative
    assert "src/manifest.py" in relative


def test_operating_year_segments_and_boundary_patch() -> None:
    segments = runner._segments(2024)
    assert len(segments["full"]) == 8784
    assert segments["full"][-1] == runner.BOUNDARY
    valid = runner.YEAR_2024[runner.YEAR_2024 != runner.BOUNDARY]
    assert len(valid) == 8783
    assert valid[-1] == pd.Timestamp("2024-12-31 23:00")


def test_stage1_baseline_reconstruction_contract() -> None:
    v1, _, _, _, _ = runner.verify_config(
        ROOT / "configs/jma_gsm_paired_increment_preregister_v5.json"
    )
    baseline = runner._load_stage1_baseline(v1)
    assert baseline.index.equals(runner.YEAR_2023)
    assert baseline.loc[:, ["kpx_group_1", "kpx_group_2"]].notna().all().all()
    assert baseline.loc[runner.G3_H1, "kpx_group_3"].isna().all()
    assert baseline.loc[runner.G3_H2, "kpx_group_3"].notna().all()


def test_existing_output_directory_is_rejected_before_materialization(tmp_path: Path) -> None:
    out = tmp_path / "already-exists"
    out.mkdir()
    args = runner.parse_args(["--stage", "stage1", "--out-dir", str(out)])
    v1, v2, v3, v4, v5 = runner.verify_config(args.config)
    with pytest.raises(FileExistsError):
        runner.run_stage1(args, v1, v2, v3, v4, v5)
    assert list(out.iterdir()) == []


def test_stage1_fit_reader_materializes_only_registered_group_cells() -> None:
    args = runner.parse_args(["--stage", "stage1"])
    _, _, v3, _, _ = runner.verify_config(args.config)
    labels, evidence = runner._read_stage1_fit_labels(args, v3)
    assert labels["kpx_group_1"].index.equals(runner.YEAR_2022)
    assert labels["kpx_group_2"].index.equals(runner.YEAR_2022)
    assert labels["kpx_group_3"].index.equals(runner.G3_H1)
    assert evidence["g12"]["columns_materialized"] == ["kpx_group_1", "kpx_group_2"]
    assert evidence["g12"]["label_value_cells_materialized"] == 17520
    assert evidence["g3"]["columns_materialized"] == ["kpx_group_3"]
    assert evidence["g3"]["label_value_cells_materialized"] == 4344
    assert evidence["score_label_value_cells_materialized"] == 0


def test_score_reader_is_fail_armed_before_any_slice_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = runner.parse_args(["--stage", "stage1"])
    _, _, v3, _, _ = runner.verify_config(args.config)
    decoded = False

    def forbidden_decode(*args: object, **kwargs: object) -> object:
        nonlocal decoded
        decoded = True
        raise AssertionError("score slice decoder was reached")

    monkeypatch.setattr(runner, "_read_label_slice", forbidden_decode)
    missing = tmp_path / "missing-candidate-lock.json"
    with pytest.raises(AssertionError, match="registered file identity differs"):
        runner._read_stage1_score_labels(
            args,
            v3,
            candidate_lock={"path": str(missing), "size_bytes": 1, "sha256": "0" * 64},
        )
    assert decoded is False


def test_candidate_lock_call_precedes_score_reader_in_both_stages() -> None:
    stage1 = inspect.getsource(runner.run_stage1)
    assert stage1.index("global_candidate_lock_path") < stage1.index("_read_stage1_score_labels")
    assert stage1.index("_read_stage1_score_labels") < stage1.index("_comparison(")
    stage2 = inspect.getsource(runner.run_stage2)
    assert stage2.index("stage2_candidate_lock_path") < stage2.index("_read_stage2_score_labels")
    assert stage2.index("_read_stage2_score_labels") < stage2.index("_comparison(")


def test_stage1_JMA_reader_is_physical_2022_2023_and_full_path_fail_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = runner.parse_args(["--stage", "stage1"])
    v1, _, _, v4, _ = runner.verify_config(args.config)
    full = Path(v1["external_source"]["normalized_2022_2024"]["path"]).resolve()
    original_verify = runner._verify_file
    opened: list[Path] = []

    def guarded_verify(spec: dict[str, object], *, base: Path = runner.PROJECT_DIR) -> Path:
        path = Path(str(spec["path"]))
        resolved = (path if path.is_absolute() else base / path).resolve()
        opened.append(resolved)
        if resolved == full:
            raise AssertionError("full JMA path opened during Stage1")
        return original_verify(spec, base=base)

    monkeypatch.setattr(runner, "_verify_file", guarded_verify)
    frame = runner._read_jma(v1, v4, period="stage1")
    assert len(frame) == 52560
    assert frame.groupby("group").size().eq(17520).all()
    assert frame["time"].min() == pd.Timestamp("2022-01-01 00:00")
    assert frame["time"].max() == pd.Timestamp("2023-12-31 23:00")
    assert full not in opened


def test_stage1_boundary_is_excluded_from_both_model_predictions() -> None:
    source = inspect.getsource(runner.run_stage1)
    assert "model_apply = apply[apply != STAGE1_BOUNDARY]" in source
    assert source.count("boundary_candidate_baseline_bit_identity") == 1
    assert runner.STAGE1_BOUNDARY == pd.Timestamp("2024-01-01 00:00")


def _registered_stage1_score_frame() -> tuple[pd.DataFrame, dict[str, object]]:
    args = runner.parse_args(["--stage", "stage1"])
    _, _, v3, _, v5 = runner.verify_config(args.config)
    source = runner._verify_file(v3["label_slice_contract"]["source"], base=args.raw_dir)
    g12, _ = runner._read_label_slice(
        source,
        v3["label_slice_contract"]["stage1_g12_score"],
        expected_index=runner.YEAR_2023,
    )
    g3, _ = runner._read_label_slice(
        source,
        v3["label_slice_contract"]["stage1_g3_score"],
        expected_index=runner.G3_H2,
    )
    score = pd.DataFrame(
        np.nan, index=runner.YEAR_2023, columns=runner.TARGET_COLS, dtype=np.float64
    )
    score.loc[runner.YEAR_2023, list(runner.TARGET_COLS[:2])] = g12.to_numpy(np.float64)
    score.loc[runner.G3_H2, runner.TARGET_COLS[2]] = g3.iloc[:, 0].to_numpy(np.float64)
    return score, v5


def test_stage1_score_nonfinite_and_eligible_masks_are_exact_and_preserved() -> None:
    score, v5 = _registered_stage1_score_frame()
    before = score.copy(deep=True)
    masks, evidence = runner._verify_stage1_score_masks(score, v5)
    assert score.equals(before)
    assert [evidence[group]["nonfinite_count"] for group in runner.TARGET_COLS] == [3, 2, 0]
    assert [evidence[group]["eligible_count"] for group in runner.TARGET_COLS] == [5508, 5473, 2311]
    assert all(mask.dtype == np.bool_ and mask.any() for mask in masks.values())


def test_stage1_candidate_forecast_finite_on_eligible_is_fail_armed() -> None:
    score, v5 = _registered_stage1_score_frame()
    masks, _ = runner._verify_stage1_score_masks(score, v5)
    baseline = pd.DataFrame(0.0, index=runner.YEAR_2023, columns=runner.TARGET_COLS)
    predictions = {
        candidate: pd.DataFrame(0.0, index=runner.YEAR_2023, columns=runner.TARGET_COLS)
        for candidate in runner.CANDIDATES
    }
    runner._verify_stage1_forecasts_on_eligible_rows(score, baseline, predictions, masks)
    group = runner.TARGET_COLS[0]
    first_eligible = runner.YEAR_2023[np.flatnonzero(masks[group])[0]]
    predictions[runner.CANDIDATES[0]].loc[first_eligible, group] = np.nan
    with pytest.raises(AssertionError, match="nonfinite on eligible rows"):
        runner._verify_stage1_forecasts_on_eligible_rows(score, baseline, predictions, masks)


def test_candidate_lock_json_key_order_is_irrelevant(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "lock_kind": "stage1_candidate_before_score_labels",
        "config_v5_sha256": V5_SHA,
        "score_label_value_cells_before_lock": 0,
    }
    for index, ordered in enumerate((payload, dict(reversed(tuple(payload.items()))))):
        path = tmp_path / f"lock-{index}.json"
        path.write_text(json.dumps(ordered), encoding="utf-8")
        record = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}
        assert runner._verify_candidate_lock(
            record,
            expected_kind="stage1_candidate_before_score_labels",
            expected_config_sha=V5_SHA,
        ) == path.resolve()


def test_manifest_output_closure_excludes_manifest_itself(tmp_path: Path) -> None:
    out = tmp_path / "artifact"
    out.mkdir()
    payload = out / "payload.bin"
    payload.write_bytes(b"frozen-output")
    args = runner.parse_args(["--out-dir", str(out)])
    runner._write_manifest(args, status="TEST", closure={}, risk={})
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_count_excluding_manifest"] == 1
    assert [record["path"] for record in manifest["outputs"]] == [str(payload.resolve())]
    assert all(Path(record["path"]).name != "manifest.json" for record in manifest["outputs"])

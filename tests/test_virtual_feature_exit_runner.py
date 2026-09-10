from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_virtual_feature_exit as runner
from src.density_ratio import assemble_locked_group
from src.manifest import describe_file


def test_inner_source_lock_uses_the_bound_preregister_copy_name() -> None:
    source = inspect.getsource(runner._run_inner)
    assert '"preregister": describe_file(prereg_copy)' in source
    assert "preregister_copy" not in source


def test_frozen_contract_ledger_folds_and_q07_recipe() -> None:
    contract = runner._verify_contract(runner.DEFAULT_PREREGISTER)
    assert runner.sha256_file(runner.DEFAULT_PREREGISTER) == runner.PREREGISTER_SHA256
    assert tuple(row["id"] for row in contract["candidate_ledger_fixed_before_labels"]) == runner.VARIANTS
    folds = runner._folds(contract)
    assert [len(folds[group]) for group in runner.TARGET_COLS] == [3, 3, 4]
    for group_folds in folds.values():
        for fold in group_folds:
            assert fold.train_end < fold.valid_start
    params = runner._model_parameters(contract)
    assert params == contract["model_contract"]["parameters"]
    assert params["objective"] == "quantile"
    assert params["alpha"] == 0.7
    assert params["n_estimators"] == 1500
    assert params["n_jobs"] == 7


def test_bounded_label_reader_does_not_parse_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = (
        "kst_dtm,kpx_group_1,kpx_group_2,kpx_group_3\n"
        "2022-01-01 01:00:00,2160,2160,2100\n"
        "2022-01-01 02:00:00,2161,2161,2101\n"
    ).encode("utf-8")
    suffix = b"FUTURE_SENTINEL,not,a,number,that,must,not,be,parsed\n"
    path = tmp_path / "labels.csv"
    path.write_bytes(rows + suffix)
    digest = hashlib.sha256(rows).hexdigest()
    expected_index = pd.date_range(
        "2022-01-01 01:00:00", periods=2, freq="h", name="forecast_kst_dtm"
    )
    monkeypatch.setattr(runner, "PREFIX_ROWS", 2)
    monkeypatch.setattr(runner, "PREFIX_BYTES", len(rows))
    monkeypatch.setattr(runner, "PREFIX_SHA256", digest)
    monkeypatch.setattr(runner, "EXPECTED_PREFIX_INDEX", expected_index)
    contract = {
        "data_contract": {
            "train_label_prefix": {
                "data_rows": 2,
                "bytes": len(rows),
                "sha256": digest,
            }
        }
    }
    frame, evidence = runner._read_bounded_labels(path, contract)
    assert frame.index.equals(expected_index)
    assert evidence["bytes_returned_to_parser"] == len(rows)
    assert evidence["future_label_bytes_read"] == 0
    assert evidence["future_label_value_cells_materialized"] == 0
    assert path.read_bytes()[len(rows) :] == suffix


def _direct_score_fixture() -> dict[str, object]:
    variants: dict[str, object] = {}
    for position, variant in enumerate(runner.EXIT_VARIANTS):
        good = position == 0
        delta = 0.01 if good else -0.002 - 0.001 * position
        variants[variant] = {
            "delta_score": delta,
            "delta_N": 0.001 if good else -0.001,
            "delta_F": 0.01 if good else -0.003,
            "cell_deltas": [0.01] * 10 if good else [-0.002] * 10,
            "worst_cell": 0.01 if good else -0.002,
            "positive_cell_count": 10 if good else 0,
        }
    return {"control_inner_official_score": 0.60, "variants": variants}


def test_max_t_risk_selection_and_public_anchor_is_report_only() -> None:
    contract = runner._verify_contract(runner.DEFAULT_PREREGISTER)
    direct = _direct_score_fixture()
    observed = np.asarray(
        [direct["variants"][name]["delta_score"] for name in runner.EXIT_VARIANTS]
    )
    draws = np.repeat(observed[None, :], runner.BOOTSTRAP_REPLICATES, axis=0)
    altered = copy.deepcopy(contract)
    altered["score_claim"]["report_only_public_anchor"] = 123.0
    ledger, trajectory = runner._stress_ledger(direct, draws, altered)
    assert ledger["max_centered_c90"] == 0.0
    assert ledger["selected_for_outer"] == runner.EXIT_VARIANTS[0]
    first = trajectory["trajectory"][1]
    assert first["public_anchor_display_only"] == pytest.approx(
        123.0 + first["risk_adjusted_increment"]
    )
    altered["score_claim"]["report_only_public_anchor"] = -999.0
    second_ledger, _ = runner._stress_ledger(direct, draws, altered)
    assert second_ledger["selected_for_outer"] == ledger["selected_for_outer"]
    assert second_ledger["candidates"] == ledger["candidates"]


def test_zero_state_and_no_overwrite_guards(tmp_path: Path) -> None:
    existing_inner = tmp_path / "already_exists"
    existing_inner.mkdir()
    with pytest.raises(FileExistsError):
        runner._run_inner(
            raw_dir=tmp_path,
            artifact_root=tmp_path,
            out_dir=existing_inner,
            preregister_path=runner.DEFAULT_PREREGISTER,
        )
    root = tmp_path / "root"
    (root / "stress").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        runner._run_stress(
            raw_dir=tmp_path,
            out_dir=root,
            preregister_path=runner.DEFAULT_PREREGISTER,
        )


def test_no_candidate_outer_skips_every_data_and_component_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inner = tmp_path / "inner"
    stress = tmp_path / "stress"
    inner.mkdir()
    stress.mkdir()
    prescore_path = inner / "inner_prescore_lock.json"
    score_path = inner / "inner_direct_scores.json"
    ledger_path = stress / "bootstrap_draw_and_candidate_ledger.json"
    prescore_path.write_text("{}\n", encoding="utf-8")
    score_path.write_text("{}\n", encoding="utf-8")
    ledger_path.write_text(
        json.dumps(
            {
                "preregister_sha256": runner.PREREGISTER_SHA256,
                "observed_delta_order": list(runner.EXIT_VARIANTS),
                "replicates": runner.BOOTSTRAP_REPLICATES,
                "seed": runner.BOOTSTRAP_SEED,
                "selected_for_outer": None,
                "passing_candidates": [],
                "candidates": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    promotion = {
        "preregister_sha256": runner.PREREGISTER_SHA256,
        "inner_prescore_lock": describe_file(prescore_path),
        "inner_direct_scores": describe_file(score_path),
        "bootstrap_ledger": describe_file(ledger_path),
        "selected_for_outer": None,
    }
    (stress / "inner_promotion_lock.json").write_text(
        json.dumps(promotion) + "\n", encoding="utf-8"
    )
    promotion_path = stress / "inner_promotion_lock.json"
    (stress / "manifest.json").write_text(
        json.dumps(
            {
                "preregister_sha256": runner.PREREGISTER_SHA256,
                "outputs": [describe_file(promotion_path)],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("OUTER data/component reader must not run without selection")

    for name in (
        "_read_bounded_labels",
        "_read_weather_prefix",
        "_load_locked_recipe",
        "_load_locked_components",
        "_fit_one",
    ):
        monkeypatch.setattr(runner, name, forbidden)
    locked_bytes = promotion_path.read_bytes()
    promotion_path.write_bytes(locked_bytes + b" ")
    with pytest.raises(AssertionError, match="locked artifact changed"):
        runner._run_outer(
            raw_dir=tmp_path,
            artifact_root=tmp_path,
            out_dir=tmp_path,
            preregister_path=runner.DEFAULT_PREREGISTER,
        )
    promotion_path.write_bytes(locked_bytes)
    runner._run_outer(
        raw_dir=tmp_path / "missing_raw",
        artifact_root=tmp_path / "missing_artifacts",
        out_dir=tmp_path,
        preregister_path=runner.DEFAULT_PREREGISTER,
    )
    rejection = json.loads(
        (tmp_path / "outer/outer_lock_or_rejection.json").read_text(encoding="utf-8")
    )
    assert rejection["status"] == "rejected_before_outer"
    assert rejection["label_bytes_read_by_outer"] == 0
    assert rejection["model_fits_by_outer"] == 0
    terminal_manifest = json.loads(
        (tmp_path / "outer/manifest.json").read_text(encoding="utf-8")
    )
    assert terminal_manifest["terminal"] is True
    assert terminal_manifest["component_files_read_by_outer"] == 0
    assert terminal_manifest["submission_created"] is False
    with pytest.raises(FileExistsError):
        runner._run_outer(
            raw_dir=tmp_path,
            artifact_root=tmp_path,
            out_dir=tmp_path,
            preregister_path=runner.DEFAULT_PREREGISTER,
        )


def test_actual_locked_g12_component_reconstruction_is_bit_exact() -> None:
    contract = runner._verify_contract(runner.DEFAULT_PREREGISTER)
    recipe, _ = runner._load_locked_recipe(contract)
    index = pd.date_range(
        "2023-01-01 01:00:00",
        "2024-01-01 00:00:00",
        freq="h",
        name="forecast_kst_dtm",
    )
    baseline_path = runner.PROJECT_DIR / "artifacts/oof/dev2023_locked_v3.parquet"
    if not baseline_path.is_file():
        pytest.skip("locked OOF artifacts are not installed")
    baseline = pd.read_parquet(baseline_path, engine="pyarrow")
    baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
    for group in runner.TARGET_COLS[:2]:
        components, _ = runner._load_locked_components(
            runner.PROJECT_DIR / "artifacts", group, index
        )
        reconstructed = assemble_locked_group(
            components,
            group=group,
            capacity_kwh=runner.CAPACITY_KWH[group],
            ensemble=recipe["ensemble"],
        )
        assert np.array_equal(reconstructed, baseline[group].to_numpy(dtype=np.float64))


def test_outer_anchor_hashes_and_bit_exact_guard() -> None:
    contract = runner._verify_contract(runner.DEFAULT_PREREGISTER)
    anchors = runner._verify_outer_anchors(
        contract, runner.PROJECT_DIR / "artifacts"
    )
    assert anchors["verified_before_outer_label_prefix_parse_or_model_fit"] is True
    runner._assert_bit_exact(np.array([1.0]), np.array([1.0]), "control")
    with pytest.raises(AssertionError, match="not bit exact"):
        runner._assert_bit_exact(
            np.array([1.0]), np.array([np.nextafter(1.0, 2.0)]), "control"
        )

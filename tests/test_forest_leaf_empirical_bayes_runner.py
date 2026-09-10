from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

from scripts import run_forest_leaf_empirical_bayes as runner
from scripts import run_full_weather_residual_pmf as parent
from src.forest_leaf_empirical_bayes import FOREST_PARAMETERS
from src.metric import TARGET_COLS


def test_preregister_and_incident_are_hash_locked() -> None:
    payload = runner._verify_preregister(
        Path("configs/forest_leaf_empirical_residual_bayes_preregister_v3.json")
    )
    assert payload["experiment_id"] == runner.ARTIFACT_TYPE
    assert payload["inherits_complete_scientific_contract"]["sha256"] == (
        runner.V2_PREREGISTER_SHA256
    )
    assert payload["claims"]["selection_safe"] is False
    assert payload["claims"]["strict_private"] is False
    assert payload["incident_chain_append"]["sha256"] == (
        runner.V2_GUARD_FAILURE_SHA256
    )


def test_parent_adapter_replaces_candidate_specific_hooks() -> None:
    runner._configure_parent()
    assert parent.PREREGISTER_SHA256 == runner.PREREGISTER_SHA256
    assert parent.ARTIFACT_TYPE == runner.ARTIFACT_TYPE
    assert parent._fit_predict_locked is runner._fit_predict_locked
    assert parent._source_snapshot is runner._source_snapshot


def test_real_temp_guard_binds_v3_identity_without_label_fit_predict_or_score(
    tmp_path: Path, monkeypatch
) -> None:
    calls = {"label": 0, "fit": 0, "predict": 0, "score": 0}

    def forbidden_label_reader(*args, **kwargs):
        calls["label"] += 1
        raise AssertionError("guard regression touched label reader")

    monkeypatch.setattr(parent.strict, "_read_full_labels", forbidden_label_reader)
    path = tmp_path / "heavy_cpu_fit.pid.json"
    owner = parent.guardmod.acquire_heavy_guard_for_experiment(
        path, experiment_id=runner.ARTIFACT_TYPE
    )
    try:
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk["experiment_id"] == "forest_leaf_empirical_residual_bayes_v3"
        assert on_disk["pid"] == os.getpid() == owner["pid"]
        assert parent.guardmod._pid_is_alive(on_disk["pid"])
        assert isinstance(on_disk["token"], str) and on_disk["token"]
        assert on_disk["token"] == owner["token"]
        assert calls == {"label": 0, "fit": 0, "predict": 0, "score": 0}
    finally:
        parent.guardmod.release_heavy_guard(path, owner)
    assert not path.exists()


def test_exact_primary_delta_is_transferred_once_to_recent() -> None:
    index = pd.date_range("2024-01-01 01:00", periods=4, freq="h")
    primary = pd.DataFrame(0.0, index=index, columns=TARGET_COLS)
    recent = pd.DataFrame(0.0, index=index, columns=TARGET_COLS)
    actions = {}
    for position, group in enumerate(TARGET_COLS):
        capacity = parent.CAPACITY_KWH[group]
        primary[group] = (0.30 + 0.05 * position) * capacity
        recent[group] = (0.45 + 0.03 * position) * capacity
        actions[group] = pd.Series(
            np.asarray([-0.10, -0.05, 0.05, 0.10]), index=index
        )
    primary_candidate, recent_candidate, delta = parent._same_primary_delta_candidates(
        primary, recent, actions
    )
    for group in TARGET_COLS:
        capacity = parent.CAPACITY_KWH[group]
        expected_delta = 0.10 * actions[group].to_numpy()
        assert np.allclose(delta[group], expected_delta, atol=1e-15, rtol=0.0)
        assert np.allclose(
            recent_candidate[group].to_numpy() / capacity,
            recent[group].to_numpy() / capacity + expected_delta,
            atol=1e-15,
            rtol=0.0,
        )
        assert np.allclose(
            primary_candidate[group].to_numpy() / capacity,
            primary[group].to_numpy() / capacity + expected_delta,
            atol=1e-15,
            rtol=0.0,
        )

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import record_public_scale_092_095_feedback_hybrid as recorder
from src.manifest import sha256_file


def _triplet(base: dict[str, Decimal], shift: str) -> dict[str, str]:
    delta = Decimal(shift)
    return {name: format(base[name] + delta, "f") for name in recorder.COMPONENTS}


def _feedback(
    config: dict,
    shifts: dict[tuple[str, str], str],
) -> dict:
    base = {
        name: Decimal(config["metric_contract"]["base"][name])
        for name in recorder.COMPONENTS
    }
    rows = []
    for factor in recorder.FACTORS:
        for group in recorder.OBSERVED_GROUPS:
            spec = config["canonical_csvs"]["group_factor"][factor][group]
            rows.append(
                {
                    "factor": factor,
                    "group": group,
                    "file": str((recorder.PROJECT_ROOT / spec["path"]).resolve()),
                    "sha256": spec["sha256"],
                    **_triplet(base, shifts[(factor, group)]),
                }
            )
    return {"schema_version": 1, "group_only_results": rows}


def _write_feedback(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_preregister_is_frozen_and_actual_state_is_zero() -> None:
    config = recorder._load_config()
    assert sha256_file(recorder.CONFIG_PATH) == recorder.CONFIG_SHA256
    assert config["pre_feedback_state"]["selection_values_materialized"] == 0
    assert config["pre_feedback_state"]["selected_groups_or_factors"] == []
    assert config["pre_feedback_state"]["hybrid_csv_bytes"] == 0
    assert not recorder.DEFAULT_FEEDBACK.exists()
    assert not recorder.DEFAULT_OUT_DIR.exists()


def test_incomplete_feedback_cannot_reach_selection_or_create_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = recorder._load_config()
    payload = _feedback(
        config,
        {
            ("0.95", "kpx_group_1"): "0.001",
            ("0.95", "kpx_group_3"): "-0.0002",
            ("0.92", "kpx_group_1"): "-0.001",
            ("0.92", "kpx_group_3"): "0.002",
        },
    )
    payload["group_only_results"].pop()
    feedback = tmp_path / "feedback.json"
    out_dir = tmp_path / "out"
    _write_feedback(feedback, payload)
    called = 0

    def forbidden(*args, **kwargs):
        nonlocal called
        called += 1
        raise AssertionError("selection must not run")

    monkeypatch.setattr(recorder, "infer_and_select", forbidden)
    with pytest.raises(ValueError, match="exactly four"):
        recorder.run(feedback, out_dir, synthetic=True)
    assert called == 0
    assert not out_dir.exists()


def test_feedback_artifact_identity_is_strict(tmp_path: Path) -> None:
    config = recorder._load_config()
    payload = _feedback(
        config,
        {
            ("0.95", "kpx_group_1"): "0.001",
            ("0.95", "kpx_group_3"): "-0.0002",
            ("0.92", "kpx_group_1"): "-0.001",
            ("0.92", "kpx_group_3"): "0.002",
        },
    )
    payload["group_only_results"][0]["sha256"] = "0" * 64
    feedback = tmp_path / "feedback.json"
    out_dir = tmp_path / "out"
    _write_feedback(feedback, payload)
    with pytest.raises(ValueError, match="artifact identity"):
        recorder.run(feedback, out_dir, synthetic=True)
    assert not out_dir.exists()


def test_decimal_inference_selection_and_synthetic_e2e(tmp_path: Path) -> None:
    config = recorder._load_config()
    payload = _feedback(
        config,
        {
            ("0.95", "kpx_group_1"): "0.001",
            ("0.95", "kpx_group_3"): "-0.0002",
            ("0.92", "kpx_group_1"): "-0.001",
            ("0.92", "kpx_group_3"): "0.002",
        },
    )
    feedback = tmp_path / "feedback.json"
    out_dir = tmp_path / "out"
    _write_feedback(feedback, payload)
    result = recorder.run(feedback, out_dir, synthetic=True)
    assert result["selected_factor_by_group"] == {
        "kpx_group_1": "0.95",
        "kpx_group_2": "identity",
        "kpx_group_3": "0.92",
    }
    decision = json.loads((out_dir / "decision.json").read_text(encoding="utf-8"))
    assert decision["decision"]["decimal_additivity_exact"] is True
    assert decision["selection_unsafe"] is True
    assert decision["private_risk"] == "HIGH"
    csv_path = Path(result["hybrid_csv"])
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    output = pd.read_csv(csv_path, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    base, groups, _ = recorder.load_canonical_texts(config)
    expected = {
        "kpx_group_1": groups["0.95"]["kpx_group_1"],
        "kpx_group_2": base,
        "kpx_group_3": groups["0.92"]["kpx_group_3"],
    }
    assert tuple(output.columns) == recorder.SCHEMA
    assert len(output) == 8760
    assert output.loc[:, list(recorder.SCHEMA[:2])].equals(base.loc[:, list(recorder.SCHEMA[:2])])
    for group, source in expected.items():
        assert output[group].equals(source[group])
        assert np.array_equal(
            output[group].astype(np.float64).to_numpy().view(np.uint64),
            source[group].astype(np.float64).to_numpy().view(np.uint64),
        )
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic"] is True
    assert manifest["selection_after_all_four_feedback_rows_only"] is True
    assert manifest["duplicate_guard_passed"] is True
    assert manifest["submission_performed"] is False


def test_duplicate_global_hybrid_is_quarantined_and_not_promoted(tmp_path: Path) -> None:
    config = recorder._load_config()
    payload = _feedback(
        config,
        {
            ("0.95", "kpx_group_1"): "0.0002",
            ("0.95", "kpx_group_3"): "0.0002",
            ("0.92", "kpx_group_1"): "-0.001",
            ("0.92", "kpx_group_3"): "-0.001",
        },
    )
    feedback = tmp_path / "feedback.json"
    out_dir = tmp_path / "out"
    _write_feedback(feedback, payload)
    with pytest.raises(ValueError, match="duplicate submission SHA"):
        recorder.run(feedback, out_dir, synthetic=True)
    assert not out_dir.exists()
    quarantines = list(tmp_path.glob("_failed_out_*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / config["output_contract"]["hybrid_csv_name"]).is_file()


def test_existing_output_and_non_temp_synthetic_paths_are_refused(tmp_path: Path) -> None:
    config = recorder._load_config()
    payload = _feedback(
        config,
        {
            ("0.95", "kpx_group_1"): "0.001",
            ("0.95", "kpx_group_3"): "-0.0002",
            ("0.92", "kpx_group_1"): "-0.001",
            ("0.92", "kpx_group_3"): "0.002",
        },
    )
    feedback = tmp_path / "feedback.json"
    out_dir = tmp_path / "out"
    _write_feedback(feedback, payload)
    out_dir.mkdir()
    with pytest.raises(FileExistsError):
        recorder.run(feedback, out_dir, synthetic=True)
    with pytest.raises(ValueError, match="OS temp"):
        recorder._assert_synthetic_paths(recorder.CONFIG_PATH, tmp_path / "x")

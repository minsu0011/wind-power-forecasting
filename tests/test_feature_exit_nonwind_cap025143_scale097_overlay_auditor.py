from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

from scripts import audit_feature_exit_nonwind_cap025143_scale097_overlay as auditor


class _FakeEstimator:
    def __init__(self, parameters: dict[str, object]) -> None:
        self._parameters = parameters

    def get_params(self) -> dict[str, object]:
        return dict(self._parameters)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(frame), dtype=np.float64)


def _target_frame(index: pd.DatetimeIndex, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {group: values * auditor.CAPACITY_KWH[group] for group in auditor.TARGET_COLS},
        index=index,
        dtype=np.float64,
    )


def _synthetic_ensemble() -> dict[str, object]:
    weights = {
        group: {
            name: float(name == "lgb_q07") for name in auditor.COMPONENT_NAMES
        }
        for group in auditor.TARGET_COLS
    }
    affine = {
        group: {"scale": 1.0, "bias_kwh": 0.0}
        for group in auditor.TARGET_COLS
    }
    clip = {
        group: {"lower_capacity_fraction": 0.0, "upper_capacity_fraction": 1.02}
        for group in auditor.TARGET_COLS
    }
    power_bins: dict[str, object] = {
        "kpx_group_1": None,
        "kpx_group_2": {
            "edges_cf": [0.0, 0.25, 0.50, 1.02],
            "delta_kwh": [0.0, 100.0, 200.0],
        },
        "kpx_group_3": None,
    }
    return {
        "weights": weights,
        "affine": affine,
        "clip": clip,
        "power_bins": power_bins,
    }


def test_static_audit_reads_only_text_contracts() -> None:
    report = auditor.static_audit(auditor.ROOT)

    assert report["status"] == "PASS_STATIC"
    assert all(report["checks"].values())
    ledger = report["access_ledger"]
    assert ledger["config_json_files_read"] == 3
    assert ledger["sidecar_text_files_read"] == 3
    for key in (
        "label_bytes_read",
        "parquet_arrays_read",
        "model_files_loaded",
        "test_arrays_read",
        "csv_files_read",
        "files_written",
        "model_fits",
    ):
        assert ledger[key] == 0


def test_cap_is_applied_once_before_q1_outer_clip() -> None:
    index = pd.date_range("2023-01-01 01:00:00", periods=3, freq="h")
    canonical = _target_frame(index, np.array([0.30, 0.01, 1.015]))
    control = _target_frame(index, np.repeat(0.50, 3))
    exit_cf = control.copy()
    raw_delta = np.array([2.0 * auditor.CAP_CF, -2.0 * auditor.CAP_CF, 0.01])
    for group in auditor.TARGET_COLS:
        # Paired model outputs are capacity fractions, not kWh.
        control[group] = 0.50
        exit_cf[group] = 0.50 + raw_delta

    result = auditor.cap_q07_component(canonical, control, exit_cf)

    for group in auditor.TARGET_COLS:
        capacity = auditor.CAPACITY_KWH[group]
        expected_capped = np.clip(
            result.raw_delta_cf[group].to_numpy(),
            -auditor.CAP_CF,
            auditor.CAP_CF,
        )
        assert np.array_equal(
            result.capped_delta_cf[group].to_numpy(), expected_capped
        )
        expected_q1 = capacity * np.clip(
            canonical[group].to_numpy() / capacity + expected_capped,
            0.0,
            1.02,
        )
        assert np.array_equal(result.q1_kwh[group].to_numpy(), expected_q1)
        assert result.cap_hit_count[group] == 2
        assert result.q1_outer_clip_count[group] == 2


def test_g2_freezes_original_b0_and_reports_hypothetical_crossing() -> None:
    index = pd.date_range("2023-02-01 01:00:00", periods=2, freq="h")
    components = {
        name: _target_frame(index, np.zeros(2)) for name in auditor.COMPONENT_NAMES
    }
    components["lgb_q07"] = _target_frame(index, np.array([0.24, 0.49]))
    q1 = components["lgb_q07"].copy()
    q1["kpx_group_2"] = auditor.CAPACITY_KWH["kpx_group_2"] * np.array(
        [0.26, 0.47]
    )

    result = auditor.replay_locked_assembly(
        components, q1, ensemble=_synthetic_ensemble()
    )

    assert np.array_equal(result.g2_b0, np.array([0, 1]))
    assert np.array_equal(result.g2_hypothetical_b1, np.array([1, 1]))
    assert result.g2_hypothetical_crossing_count == 1
    # First row crossed the 0.25 boundary, but A1 uses b0's zero-kWh delta.
    assert result.a1_safe_kwh.iloc[0]["kpx_group_2"] == pytest.approx(
        0.26 * auditor.CAPACITY_KWH["kpx_group_2"]
    )
    hypothetical_b1_value = (
        0.26 * auditor.CAPACITY_KWH["kpx_group_2"] + 100.0
    )
    assert result.a1_safe_kwh.iloc[0]["kpx_group_2"] != hypothetical_b1_value
    assert np.array_equal(
        result.increment_d_kwh.to_numpy(),
        (result.a1_safe_kwh - result.a0_kwh).to_numpy(),
    )


def test_final_formula_adds_d_then_clips_once() -> None:
    index = pd.date_range("2025-01-01 01:00:00", periods=3, freq="h")
    base = _target_frame(index, np.array([0.01, 0.50, 1.01]))
    delta = _target_frame(index, np.array([-0.02, 0.03, 0.03]))

    result = auditor.final_base_plus_increment(base, delta)

    for group in auditor.TARGET_COLS:
        capacity = auditor.CAPACITY_KWH[group]
        expected = capacity * np.array([0.0, 0.53, 1.02])
        assert np.array_equal(result[group].to_numpy(), expected)


def test_exact_seven_veto_metrics_are_all_positive() -> None:
    index = auditor.STRESS_INDEX
    actual = _target_frame(index, np.repeat(0.50, len(index)))
    base = actual.copy()
    candidate = actual.copy()
    for group in auditor.TARGET_COLS:
        capacity = auditor.CAPACITY_KWH[group]
        base[group] = actual[group] + 0.07 * capacity
        candidate[group] = actual[group] + 0.05 * capacity

    result = auditor.replay_seven_veto_metrics(actual, base, candidate)

    assert tuple(result["slices"]) == auditor.SLICE_ORDER
    assert result["passed"] is True
    assert result["gates"] == {
        "all_seven_score_positive": True,
        "full_n_nonnegative": True,
        "full_f_positive": True,
    }
    assert all(
        result["slices"][name]["delta"]["score"] > 0.0
        for name in auditor.SLICE_ORDER
    )


def test_csv_contract_bom_lf_six_decimals_bounds_and_roundtrip(tmp_path: Path) -> None:
    rows = len(auditor.TEST_INDEX)
    sample = pd.DataFrame(
        {
            "kst_dtm": auditor.TEST_INDEX.strftime("%Y-%m-%d %H:%M:%S"),
            **{group: "0" for group in auditor.TARGET_COLS},
        },
        dtype="string",
    )
    sample_path = tmp_path / "sample_submission.csv"
    sample.to_csv(
        sample_path,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    prediction = pd.DataFrame(
        {
            group: np.linspace(0.0, 1.02 * auditor.CAPACITY_KWH[group], rows)
            for group in auditor.TARGET_COLS
        },
        index=auditor.TEST_INDEX,
    )
    expected = sample.copy()
    for group in auditor.TARGET_COLS:
        expected[group] = [f"{value:.6f}" for value in prediction[group]]
    buffer = io.StringIO(newline="")
    expected.to_csv(buffer, index=False, lineterminator="\n")
    csv_path = tmp_path / "submission.csv"
    csv_path.write_bytes(b"\xef\xbb\xbf" + buffer.getvalue().encode("utf-8"))
    sample_sha = hashlib.sha256(sample_path.read_bytes()).hexdigest()

    result = auditor.validate_submission_csv(
        csv_path,
        sample_path,
        prediction,
        expected_sample_sha256=sample_sha,
    )

    assert result["pass"] is True
    assert all(result["checks"].values())


def test_audit_json_writer_is_strictly_no_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "audit.json"
    auditor._atomic_no_overwrite_json(destination, {"status": "first"})
    original = destination.read_bytes()

    with pytest.raises(FileExistsError):
        auditor._atomic_no_overwrite_json(destination, {"status": "second"})

    assert destination.read_bytes() == original
    assert json.loads(original) == {"status": "first"}


@pytest.mark.parametrize("promoted", [False, True])
def test_flat_v3_access_events_and_ranges_are_exact(promoted: bool) -> None:
    names = [
        "bounded_label_prefix_read",
        "source_lock_durable",
        "bounded_label_prefix_parsed_after_source_lock",
        "original_A0_reconstruction_verified_before_fit",
        "six_stress_models_double_reload_verified",
        "stress_candidate_lock_durable",
        "candidate_models_predictions_and_lock_reopened_rehashed",
        "label_suffix_single_read_and_in_memory_full_hash",
        "historical_veto_scored",
    ]
    if promoted:
        names.extend(
            [
                "promotion_and_stress_manifest_durable",
                "conditional_official_test_raw_test_caches_and_final_inputs_verified_and_read",
                "six_final_models_double_reload_verified",
                "final_csv_byte_roundtrip_verified",
            ]
        )
    events: list[dict[str, object]] = []
    for sequence, name in enumerate(names, start=1):
        event: dict[str, object] = {"sequence": sequence, "event": name}
        if sequence <= 7:
            event["suffix_reads_so_far"] = 0
        else:
            event["suffix_reads_total"] = 1
        events.append(event)
    events[0]["bytes"] = auditor.PREFIX_BYTES
    events[2]["rows"] = 17_520
    events[3]["stress_model_fits_so_far"] = 0
    events[4]["stress_model_fits"] = 6
    events[7]["bytes"] = auditor.SUFFIX_BYTES
    events[8]["passed"] = promoted
    if promoted:
        events[11]["final_model_fits"] = 6
    ledger = {
        "events": events,
        "prefix_range_disk_reads": 1,
        "suffix_range_disk_reads": 1,
        "whole_file_single_stream_reads": 0,
        "full_file_identity_computed_from_two_in_memory_ranges": True,
        "stress_model_fits": 6,
        "final_model_fits": 6 if promoted else 0,
        "test_cache_files_opened": 3 if promoted else 0,
        "final_component_files_opened": 6 if promoted else 0,
        "sample_files_opened": 1 if promoted else 0,
        "csv_files_written": 1 if promoted else 0,
        "suffix_buffer_reused_for_final_fit": promoted,
    }
    source_lock = {
        "label_prefix": {
            "bytes": auditor.PREFIX_BYTES,
            "sha256": auditor.PREFIX_SHA256,
            "suffix_bytes_read": 0,
        },
        "deferred_full_label_sha": auditor.FULL_LABEL_SHA256,
    }
    score_lock = {
        "bytes": auditor.SUFFIX_BYTES,
        "sha256": auditor.SUFFIX_SHA256,
        "disk_reads": 1,
        "rows": len(auditor.STRESS_INDEX),
        "full_sha_from_memory": auditor.FULL_LABEL_SHA256,
        "physical_file_size_bytes": auditor.FULL_LABEL_BYTES,
        "full_file_size_exact": True,
    }

    result = auditor.validate_v3_access_ledger(
        ledger,
        promoted=promoted,
        source_lock=source_lock,
        score_label_access=score_lock,
    )

    assert result["pass"] is True
    assert len(auditor.STRESS_INDEX) == 8_784
    assert len(auditor.TEST_INDEX) == 8_760


def test_double_reload_checks_payload_record_and_feature_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parameters = {"objective": "quantile", "n_jobs": 7}
    control_features = ["wind", "temperature"]
    exit_features = ["temperature"]
    feature_counts = {"control_all": 2, "nonwind_atmospheric": 1}
    feature_shas = {
        "control_all": auditor._ordered_names_sha256(control_features),
        "nonwind_atmospheric": auditor._ordered_names_sha256(exit_features),
    }
    eligible_counts = {group: position + 3 for position, group in enumerate(auditor.TARGET_COLS)}
    monkeypatch.setattr(auditor, "MODEL_PARAMS", parameters)
    monkeypatch.setattr(auditor, "FEATURE_COUNTS", feature_counts)
    monkeypatch.setattr(auditor, "CONTROL_FEATURE_SHA256", feature_shas["control_all"])
    monkeypatch.setattr(
        auditor, "NONWIND_FEATURE_SHA256", feature_shas["nonwind_atmospheric"]
    )
    monkeypatch.setattr(auditor, "STAGE_ELIGIBLE_COUNTS", {"stress": eligible_counts})
    records: list[dict[str, object]] = []
    expected_identities = {
        group: {
            "eligible_count": eligible_counts[group],
            "digest": hashlib.sha256(group.encode()).hexdigest(),
        }
        for group in auditor.TARGET_COLS
    }
    apply_index = pd.date_range("2024-01-01 01:00:00", periods=3, freq="h")
    apply_features = {
        group: pd.DataFrame(
            {"wind": np.arange(3, dtype=np.float32), "temperature": 1.0},
            index=apply_index,
        )
        for group in auditor.TARGET_COLS
    }
    zero_predictions = pd.DataFrame(
        0.0, index=apply_index, columns=auditor.TARGET_COLS
    )
    expected_predictions = {
        "control_all": zero_predictions,
        "nonwind_atmospheric": zero_predictions.copy(),
    }
    for group in auditor.TARGET_COLS:
        for raw_kind, variant, features in (
            ("control", "control_all", control_features),
            ("nonwind", "nonwind_atmospheric", exit_features),
        ):
            digest = expected_identities[group]["digest"]
            feature_sha = feature_shas[variant]
            payload = {
                "group": group,
                "kind": raw_kind,
                "features": features,
                "feature_names_sha256": feature_sha,
                "eligible_count": eligible_counts[group],
                "eligible_index_target_digest": digest,
                "parameters": parameters,
                "model": _FakeEstimator(parameters),
            }
            path = tmp_path / f"{group}__{raw_kind}.joblib"
            joblib.dump(payload, path)
            records.append(
                {
                    "group": group,
                    "kind": raw_kind,
                    "file": auditor.file_record(path),
                    "feature_count": len(features),
                    "feature_names_sha256": feature_sha,
                    "eligible_count": eligible_counts[group],
                    "eligible_index_target_digest": digest,
                    "parameters": parameters,
                    "reload_1_equal": True,
                    "reload_2_equal": True,
                }
            )

    result = auditor.audit_double_reload_records(
        records,
        project_root=tmp_path,
        stage="stress",
        expected_eligible_identities=expected_identities,
        apply_features=apply_features,
        expected_predictions=expected_predictions,
    )

    assert result["pass"] is True
    assert result["model_count"] == 6
    tampered = {group: dict(value) for group, value in expected_identities.items()}
    tampered["kpx_group_1"]["digest"] = "0" * 64
    with pytest.raises(auditor.AuditFailure, match="payload/record contract"):
        auditor.audit_double_reload_records(
            records,
            project_root=tmp_path,
            stage="stress",
            expected_eligible_identities=tampered,
            apply_features=apply_features,
            expected_predictions=expected_predictions,
        )
    wrong_predictions = {
        key: value.copy() for key, value in expected_predictions.items()
    }
    wrong_predictions["control_all"].iloc[0, 0] = 1.0
    with pytest.raises(auditor.AuditFailure, match="saved CF"):
        auditor.audit_double_reload_records(
            records,
            project_root=tmp_path,
            stage="stress",
            expected_eligible_identities=expected_identities,
            apply_features=apply_features,
            expected_predictions=wrong_predictions,
        )


def test_prediction_reader_rejects_extra_columns(tmp_path: Path) -> None:
    index = pd.date_range("2024-01-01 01:00:00", periods=2, freq="h")
    frame = _target_frame(index, np.array([0.1, 0.2]))
    frame["unexpected"] = 1.0
    path = tmp_path / "prediction.parquet"
    frame.to_parquet(path)

    with pytest.raises(auditor.AuditFailure, match="schema/order"):
        auditor._read_target_frame(path, expected_index=index)


def test_label_size_preflight_rejects_appended_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.csv"
    path.write_bytes(b"abc")
    monkeypatch.setattr(auditor, "FULL_LABEL_BYTES", 3)
    assert auditor._require_exact_label_file_size(path) == 3

    path.write_bytes(b"abcX")
    with pytest.raises(auditor.AuditFailure, match="changed/appended"):
        auditor._require_exact_label_file_size(path)


def test_canonical_q07_replay_is_bit_exact_and_checks_manifest_claim(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "canonical_q07.joblib"
    payload = {
        "models": {
            group: _FakeEstimator({"objective": "quantile"})
            for group in auditor.TARGET_COLS
        },
        "feature_names": {group: ["x"] for group in auditor.TARGET_COLS},
    }
    joblib.dump(payload, model_path)
    prediction_path = tmp_path / "canonical_q07.parquet"
    stored = pd.DataFrame(
        0.0, index=auditor.TEST_INDEX, columns=auditor.TARGET_COLS
    )
    stored.to_parquet(prediction_path)
    science = {
        "canonical_component_contract": {
            "canonical_q07_model": auditor.file_record(model_path),
            "canonical_q07_prediction": auditor.file_record(prediction_path),
        }
    }
    features = {
        group: pd.DataFrame({"x": 1.0}, index=auditor.TEST_INDEX)
        for group in auditor.TARGET_COLS
    }
    claim = {
        group: {"array_equal": True, "max_abs_kwh": 0.0}
        for group in auditor.TARGET_COLS
    }

    result = auditor.audit_canonical_q07_replay(
        science_v1=science,
        project_root=tmp_path,
        test_features=features,
        manifest_claim=claim,
    )

    assert result["manifest_claim_exact"] is True
    wrong_claim = {group: dict(value) for group, value in claim.items()}
    wrong_claim["kpx_group_2"]["max_abs_kwh"] = 1.0
    with pytest.raises(auditor.AuditFailure, match="metric float differs"):
        auditor.audit_canonical_q07_replay(
            science_v1=science,
            project_root=tmp_path,
            test_features=features,
            manifest_claim=wrong_claim,
        )


def test_source_lock_rejects_any_verified_input_omission() -> None:
    science_v1, _, _ = auditor.load_effective_science_head(auditor.ROOT)
    exact = auditor.expected_verified_source_inputs(auditor.ROOT, science_v1)

    assert len(exact) == 29
    assert all(record["hash_verified"] is True for record in exact)
    with pytest.raises(auditor.AuditFailure, match="set/order/count"):
        auditor.audit_verified_source_inputs(
            exact[:-1], project_root=auditor.ROOT, science_v1=science_v1
        )
    reordered = exact.copy()
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(auditor.AuditFailure, match="set/order/count"):
        auditor.audit_verified_source_inputs(
            reordered, project_root=auditor.ROOT, science_v1=science_v1
        )


def test_execution_lineage_copy_and_review_binding_are_fail_closed(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.json"
    copied = tmp_path / "copied.json"
    canonical.write_bytes(b'{"locked":true}\n')
    copied.write_bytes(canonical.read_bytes())
    auditor._require_bit_exact_copy(copied, canonical)
    copied.write_bytes(b'{"locked":false}\n')
    with pytest.raises(auditor.AuditFailure, match="lineage differs"):
        auditor._require_bit_exact_copy(copied, canonical)

    amendment_sha = "a" * 64
    runner_sha = "b" * 64
    review = {
        "experiment_id": "feature_exit_nonwind_cap025143_scale097_overlay_v1",
        "verdict": "PASS",
        "blocking_defects": 0,
        "execution_amendment": {"sha256": amendment_sha},
        "bound_runner": {"sha256": runner_sha},
    }
    auditor._require_review_binding(
        review, amendment_sha=amendment_sha, runner_sha=runner_sha
    )
    review["bound_runner"]["sha256"] = "c" * 64
    with pytest.raises(auditor.AuditFailure, match="bind amendment/runner"):
        auditor._require_review_binding(
            review, amendment_sha=amendment_sha, runner_sha=runner_sha
        )


def test_attempt_lock_requires_exact_source_root_canonical_closure(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "attempt.json"
    canonical.write_text(
        json.dumps(
            {
                "experiment_id": (
                    "feature_exit_nonwind_cap025143_scale097_overlay_v1"
                ),
                "single_attempt": True,
            }
        ),
        encoding="utf-8",
    )
    record = auditor.file_record(canonical)
    result = auditor._audit_attempt_closure(
        record,
        dict(record),
        canonical_attempt=canonical.resolve(),
        project_root=tmp_path,
    )
    assert result == record

    tampered_root = dict(record)
    tampered_root["sha256"] = "0" * 64
    with pytest.raises(auditor.AuditFailure, match="source/root"):
        auditor._audit_attempt_closure(
            record,
            tampered_root,
            canonical_attempt=canonical.resolve(),
            project_root=tmp_path,
        )

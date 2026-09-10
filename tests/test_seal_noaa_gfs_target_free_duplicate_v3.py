from __future__ import annotations

import ast
import copy
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts import seal_noaa_gfs_target_free_duplicate_v3 as seal


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def amendment() -> dict[str, Any]:
    # Immutable JSON/schema metadata only: no Parquet value access is performed.
    return seal.validate_amendment(seal.ARTIFACT_ROOT_DEFAULT)


def _make_code_repo(root: Path) -> dict[str, dict[str, Any]]:
    for ordinal, relative in enumerate(seal.CODE_ROLE_PATHS.values(), start=1):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-role-{ordinal}\n".encode("ascii"))
    return seal.collect_code_identities(root)


def _test_evidence(repo: Path, amendment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "commands": seal.expected_test_commands(repo, amendment),
        "results": {
            role: {
                "test_file": seal.TEST_ROLE_PATHS[role],
                "exit_code": 0,
                "passed": 1,
                "skipped": 0,
                "deselected": 0,
                "summary": "1 passed in 0.01s",
            }
            for role in ("runner", "auditor", "sealer")
        },
        "all_exit_codes_zero": True,
        "all_expected_test_files_covered": True,
        "network_denied": True,
        "pycache_disabled": True,
        "code_identities_revalidated": True,
        "completed_utc": "2026-08-11T16:00:00.000000Z",
    }


def _identity(path: str, marker: str = "a", size: int = 1) -> dict[str, Any]:
    return {"path": path, "size_bytes": size, "sha256": marker * 64}


def _staged_identities() -> dict[str, Any]:
    aliases = {
        name: {
            "path": spec["final"],
            "size_bytes": spec["size_bytes"],
            "sha256": spec["sha256"],
        }
        for name, spec in seal.ALIAS_SPECS.items()
    }
    runner: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(seal.RUNNER_PUBLICATION_ORDER, start=1):
        physical = f"{index:064x}"
        logical = physical if seal.EXPECTED_RUNNER_FORMATS[name] != "PARQUET" else f"{index + 20:064x}"
        runner[name] = {
            "path": seal.RUNNER_STAGED_RELATIVES[name],
            "size_bytes": index,
            "sha256": physical,
            "format": seal.EXPECTED_RUNNER_FORMATS[name],
            "row_count": seal.EXPECTED_RUNNER_ROWS[name],
            "logical_sha256": logical,
        }
    return {"aliases": aliases, "runner_staged_outputs": runner}


def _threadpool_observation(amendment: Mapping[str, Any]) -> list[dict[str, Any]]:
    native = amendment["runtime_identity"]["native_libraries_exact"]
    return [
        {
            "name": name,
            "path": native[name]["path"],
            "user_api": native[name]["user_api"],
            "internal_api": native[name]["internal_api"],
            "prefix": native[name]["prefix"],
            "version": native[name]["version"],
            "threading_layer": native[name].get("threading_layer"),
            "architecture": native[name].get("architecture"),
            "num_threads": 1,
        }
        for name in seal.NATIVE_LIBRARY_ORDER
    ]


def _zero_fit_threadpool(amendment: Mapping[str, Any]) -> dict[str, Any]:
    event = {
        "event_ordinal": 1,
        "phase": "BEFORE_EXPLICIT_CONTEXT",
        "label": seal.THREADPOOL_BEFORE_LABEL,
        "threadpools": _threadpool_observation(amendment),
    }
    encoded = seal.compact_json_bytes(event)
    chain = hashlib.sha256()
    chain.update(len(encoded).to_bytes(8, "big"))
    chain.update(encoded)
    return {
        "capture_phases": list(seal.THREADPOOL_PHASES),
        "capture_counts": {
            "BEFORE_EXPLICIT_CONTEXT": 1,
            "INSIDE_CONTEXT": 0,
            "AFTER_CONTEXT": 0,
        },
        "event_count": 1,
        "event_chain_sha256": chain.hexdigest(),
        "first_observation_by_phase": {
            "BEFORE_EXPLICIT_CONTEXT": event,
            "INSIDE_CONTEXT": None,
            "AFTER_CONTEXT": None,
        },
        "last_observation_by_phase": {
            "BEFORE_EXPLICIT_CONTEXT": event,
            "INSIDE_CONTEXT": None,
            "AFTER_CONTEXT": None,
        },
    }


def _threadpool_for_labels(
    amendment: Mapping[str, Any], labels: list[str]
) -> dict[str, Any]:
    observations = _threadpool_observation(amendment)
    chain = hashlib.sha256()
    counts = {phase: 0 for phase in seal.THREADPOOL_PHASES}
    first = {phase: None for phase in seal.THREADPOOL_PHASES}
    last = {phase: None for phase in seal.THREADPOOL_PHASES}
    ordinal = 0

    def capture(phase: str, label: str) -> None:
        nonlocal ordinal
        ordinal += 1
        counts[phase] += 1
        event = {
            "event_ordinal": ordinal,
            "phase": phase,
            "label": label,
            "threadpools": observations,
        }
        encoded = seal.compact_json_bytes(event)
        chain.update(len(encoded).to_bytes(8, "big"))
        chain.update(encoded)
        if first[phase] is None:
            first[phase] = event
        last[phase] = event

    capture("BEFORE_EXPLICIT_CONTEXT", seal.THREADPOOL_BEFORE_LABEL)
    for label in labels:
        capture("INSIDE_CONTEXT", label)
        capture("AFTER_CONTEXT", label)
    return {
        "capture_phases": list(seal.THREADPOOL_PHASES),
        "capture_counts": counts,
        "event_count": ordinal,
        "event_chain_sha256": chain.hexdigest(),
        "first_observation_by_phase": first,
        "last_observation_by_phase": last,
    }


def _runner_stdout(
    amendment: Mapping[str, Any],
    *,
    staged: Mapping[str, Any],
    decision: Mapping[str, Any],
    attempt_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "artifact_type": "TARGET_FREE_DUPLICATE_RUNNER_STAGED_RESULT_V3",
        "status": "RUNNER_STAGED_OUTPUTS_COMPLETE_PENDING_POSTRUN_AUDIT",
        "attempt_id": attempt_id,
        "decision": dict(decision),
        "output_counts": dict(seal.AUDITOR_OUTPUT_COUNTS),
        "staged_output_identities": [
            staged["runner_staged_outputs"][name]
            for name in seal.RUNNER_PUBLICATION_ORDER
        ],
        "threadpool_info_before_inside_after": _zero_fit_threadpool(amendment),
        "no_forbidden_access_attestation": dict(
            seal.DECISION_FORBIDDEN_ACCESS_ATTESTATION
        ),
        "manifest_present": False,
        "family_lock_present": False,
    }


def _process_capture(payload: Mapping[str, Any], command: list[str]) -> dict[str, Any]:
    encoded = seal.pretty_json_bytes(payload)
    return {
        "command": command,
        "exit_code": 0,
        "stdout_payload": payload,
        "stdout_size_bytes": len(encoded),
        "stdout_sha256": _sha(encoded),
        "stderr_size_bytes": 0,
        "stderr_sha256": seal.EMPTY_SHA256,
        "wall_seconds_observed": 1.25,
        "full_stdout_identity_instrumented": True,
    }


def _auditor_stdout(
    amendment: Mapping[str, Any],
    *,
    status_key: str = "positive",
    attempt_id: str = "target_free_duplicate_v3__20260811T000000000000Z",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    statuses = amendment["output_contract"]["decision_status_literals_exact"]
    selected = (
        amendment["family_decision_and_salvage"]["fixed_priority_if_both_pass"][0]
        if status_key == "positive"
        else None
    )
    code = {
        role: _identity(f"C:/fixture/{relative}", marker=chr(97 + index))
        for index, (role, relative) in enumerate(seal.CODE_ROLE_PATHS.items())
    }
    authorization = _identity(seal.AUTHORIZATION_RELATIVE, marker="b")
    independent_go = _identity(seal.GO_RELATIVE, marker="c")
    staged = _staged_identities()
    payload = {
        "schema_version": 3,
        "artifact_type": "TARGET_FREE_DUPLICATE_NO_FIT_POSTRUN_AUDIT_V3",
        "status": "PASS_V3_NO_FIT_POSTRUN_AUDIT",
        "attempt_id": attempt_id,
        "amendment": seal.amendment_identity(),
        "amendment_v4": seal.amendment_v4_identity(),
        "authorization": authorization,
        "independent_go": independent_go,
        "code_identities": code,
        "staged_output_identities": staged,
        "output_counts": dict(seal.AUDITOR_OUTPUT_COUNTS),
        "decision": {
            "status": statuses[status_key],
            "selected_family": selected,
            "global_ambiguity": status_key == "global_ambiguity",
        },
        "poststate": dict(seal.AUDITOR_POSTSTATE),
        "files_written": 0,
        "network_requests": 0,
        "models_fit": 0,
        "forbidden_reads": dict(seal.AUDITOR_FORBIDDEN_READS),
    }
    return payload, authorization, independent_go, code


def _decision_payload(amendment: Mapping[str, Any], status_key: str) -> dict[str, Any]:
    responses = amendment["planned_fit_budget"]["response_order"]
    predictors = amendment["input_and_join_contract"]["predictor_columns_exact"]

    def raw_record(response: str, classification: str) -> dict[str, Any]:
        selector_null = classification in (
            *seal.PREMODEL_CLASSIFICATIONS,
            "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE",
        )
        ambiguous = classification in seal.AMBIGUOUS_CLASSIFICATIONS
        return {
            "response": response,
            "classification": classification,
            "counts_as_nonredundant": classification == seal.PASS_CLASSIFICATION,
            "independent_estimator": None if selector_null else "RIDGE_PIPELINE",
            "parent_reference_estimators": [] if selector_null else ["RIDGE_PIPELINE"],
            "parent_redundancy_boolean": None if classification in seal.PREMODEL_CLASSIFICATIONS or ambiguous else False,
            "affine_qualifying_predictors": (
                [predictors[0]]
                if classification == "CLEAR_REDUNDANT_AFFINE"
                else []
            ),
            "cell_stability_pass": None if classification in seal.PREMODEL_CLASSIFICATIONS or ambiguous else True,
            "distribution_stability_pass": None if classification in seal.PREMODEL_CLASSIFICATIONS or ambiguous else True,
            "boundary_equality_flags": [],
            "ambiguity_reason": classification if ambiguous else None,
        }

    raw: list[dict[str, Any]] = []
    for response in responses:
        if status_key == "positive":
            classification = (
                "CLEAR_REDUNDANT_AFFINE"
                if response == "HPBL_surface"
                else seal.PASS_CLASSIFICATION
            )
        elif status_key == "global_ambiguity" and response == "HPBL_surface":
            classification = "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
        else:
            classification = "CLEAR_REDUNDANT_AFFINE"
        raw.append(raw_record(response, classification))
    raw_by_response = {record["response"]: record for record in raw}

    endpoints: list[dict[str, Any]] = []
    endpoint_pass = status_key in ("positive", "global_ambiguity")
    for diagnostic in seal.ENDPOINT_DIAGNOSTICS:
        binding = {
            response: raw_by_response[response]["independent_estimator"]
            for response in seal.ENDPOINT_SOURCE_RESPONSES[diagnostic]
        }
        endpoints.append(
            {
                "diagnostic": diagnostic,
                "verdict": (
                    "PASS_ENDPOINT_NONREDUNDANT_MARGIN"
                    if endpoint_pass
                    else "CLEAR_ENDPOINT_VETO_FAILURE"
                ),
                "source_component_estimator_binding": binding,
                "pooled_stable_margin_pass": True if endpoint_pass else False,
                "cell_stability_pass": True,
                "distribution_stability_pass": True,
                "boundary_equality_flags": [],
                "clear_pass": endpoint_pass,
                "ambiguity_reason": None,
            }
        )

    if status_key == "positive":
        family_results = {
            "LOW_LEVEL_ISOBARIC_WIND_PROFILE": True,
            "PBL_HEIGHT": False,
        }
        selected = "LOW_LEVEL_ISOBARIC_WIND_PROFILE"
        selected_raw = [name for name in responses if name != "HPBL_surface"]
        selected_derived = list(amendment["derived_wind_contract"]["derived_order_exact"])
    else:
        family_results = {
            "LOW_LEVEL_ISOBARIC_WIND_PROFILE": False,
            "PBL_HEIGHT": False,
        }
        selected = None
        selected_raw = []
        selected_derived = []
    ties = (
        [
            {
                "tie_kind": "INDEPENDENT_RMSE",
                "diagnostic": "HPBL_surface",
                "estimators": list(seal.ESTIMATOR_ENUM),
                "metric_name": "RMSE",
                "metric_values_float_hex": [1.0.hex(), 1.0.hex()],
                "resolution": "GLOBAL_AMBIGUITY",
                "global_ambiguity": True,
            }
        ]
        if status_key == "global_ambiguity"
        else []
    )
    return {
        "schema_version": 3,
        "artifact_type": "TARGET_FREE_FAMILY_DECISION_V3",
        "status": amendment["output_contract"]["decision_status_literals_exact"][status_key],
        "raw_component_order": list(responses),
        "raw_component_results": raw,
        "endpoint_veto_results": endpoints,
        "global_ambiguity": status_key == "global_ambiguity",
        "global_ambiguity_reasons": (
            ["AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE:HPBL_surface"]
            if status_key == "global_ambiguity"
            else []
        ),
        "family_clear_pass_results": family_results,
        "fixed_priority": list(amendment["family_decision_and_salvage"]["fixed_priority_if_both_pass"]),
        "selected_family": selected,
        "selected_raw_columns": selected_raw,
        "selected_derived_columns": selected_derived,
        "all_threshold_float_hex_witnesses": [],
        "all_tied_witnesses": ties,
        "no_label_network_future_year_model_submission_attestation": dict(
            seal.DECISION_FORBIDDEN_ACCESS_ATTESTATION
        ),
    }


def _no_selector_decision_payload(
    amendment: Mapping[str, Any]
) -> dict[str, Any]:
    decision = _decision_payload(amendment, "clear_no_family")
    for record in decision["raw_component_results"]:
        record.update(
            {
                "classification": "CLEAR_NONNOVEL_CONSTANT",
                "counts_as_nonredundant": False,
                "independent_estimator": None,
                "parent_reference_estimators": [],
                "parent_redundancy_boolean": None,
                "affine_qualifying_predictors": [],
                "cell_stability_pass": None,
                "distribution_stability_pass": None,
                "boundary_equality_flags": [],
                "ambiguity_reason": None,
            }
        )
    for endpoint in decision["endpoint_veto_results"]:
        endpoint.update(
            {
                "verdict": "CLEAR_ENDPOINT_VETO_FAILURE",
                "source_component_estimator_binding": {
                    response: None
                    for response in seal.ENDPOINT_SOURCE_RESPONSES[
                        endpoint["diagnostic"]
                    ]
                },
                "pooled_stable_margin_pass": None,
                "cell_stability_pass": None,
                "distribution_stability_pass": None,
                "boundary_equality_flags": [],
                "clear_pass": False,
                "ambiguity_reason": None,
            }
        )
    return decision


def _lock_payload(amendment: Mapping[str, Any], status_key: str) -> dict[str, Any]:
    selected = (
        amendment["family_decision_and_salvage"]["fixed_priority_if_both_pass"][0]
        if status_key == "positive"
        else None
    )
    populated = [] if selected else None
    return {
        "schema_version": 3,
        "artifact_type": "TARGET_FREE_FAMILY_LOCK_V3",
        "status": amendment["output_contract"]["family_lock_status_literals_exact"][status_key],
        "selected_family": selected,
        "ordered_raw_columns": populated,
        "ordered_derived_columns": populated,
        "units": {} if selected else None,
        "grib_selectors": {} if selected else None,
        "site_group_scope": {},
        "spatial_transforms": {},
        "parent_and_independent_gate_results": {},
        "bound_input_identities": {},
        "code_config_runtime_identities": {},
        "postrun_pass_identity": _identity(seal.POSTRUN_RELATIVE),
        "no_label_attestation": True,
        "downstream_authority": amendment["output_contract"]["output_record_schemas_exact"]["TARGET_FREE_FAMILY_LOCK_json"]["downstream_authority_literal"],
    }


def _manifest_payload() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "artifact_type": "TRACK_A_TARGET_FREE_RUN_MANIFEST_V3",
        "status": "COMMITTED_V3_TARGET_FREE_DECISION",
        "terminal_state": "FIXTURE",
        "created_utc": "2026-08-11T16:00:09.000000Z",
        "code_config_runtime_identities": {},
        "control_identities": {},
        "alias_identities": {},
        "output_identities": {},
        "planned_completed_skipped_fit_counts": {},
        "oof_slot_counts": {},
        "threadpool_info_before_inside_after": {},
        "no_forbidden_access_attestation": True,
        "manifest_is_last_commit_marker": True,
    }


def _fit_ledger_payload(amendment: Mapping[str, Any]) -> dict[str, Any]:
    responses = amendment["planned_fit_budget"]["response_order"]
    predictors = amendment["input_and_join_contract"]["predictor_columns_exact"]
    slots: list[dict[str, Any]] = []
    for response_index, response in enumerate(responses, start=1):
        for fold_index, fold in enumerate(seal.FOLD_ENUM, start=1):
            for estimator_index, estimator in enumerate(seal.ESTIMATOR_ENUM, start=1):
                slots.append(
                    {
                        "planned_unit_slot_id": f"PMU__{response_index:02d}__{fold_index}__{estimator_index}",
                        "decision_fit_ordinal": len(slots) + 1,
                        "unit_kind": "PRIMARY_MODEL",
                        "response": response,
                        "fold": fold,
                        "estimator_unit": estimator,
                        "training_rows": 14_688,
                        "training_columns": 35,
                        "heldout_rows": 4_896,
                        "input_dtype": (
                            "float64_C_CONTIGUOUS"
                            if estimator == "RIDGE_PIPELINE"
                            else "float32_C_CONTIGUOUS"
                        ),
                        "response_dtype": "float64",
                        "unit_status": "SKIPPED_PREMODEL_CLEAR_VETO",
                        "skip_reason": "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3",
                        "pipeline_fit_calls": 0,
                        "standard_scaler_fit_calls": 0,
                        "ridge_fit_calls": 0,
                        "extra_trees_fit_calls": 0,
                        "predict_calls": 0,
                        "random_state": 260810 if estimator == "EXTRA_TREES" else None,
                        "fit_completed": False,
                    }
                )
    affine_ordinal = 0
    for response_index, response in enumerate(responses, start=1):
        for predictor_index, predictor in enumerate(predictors, start=1):
            for fold_index, fold in enumerate(seal.FOLD_ENUM, start=1):
                affine_ordinal += 1
                slots.append(
                    {
                        "planned_unit_slot_id": f"AAU__{response_index:02d}__{predictor_index:02d}__{fold_index}",
                        "decision_fit_ordinal": len(slots) + 1,
                        "analytic_affine_ordinal": affine_ordinal,
                        "unit_kind": "ANALYTIC_AFFINE",
                        "response": response,
                        "predictor": predictor,
                        "fold": fold,
                        "training_rows": 14_688,
                        "heldout_rows": 4_896,
                        "unit_status": "SKIPPED_PREMODEL_CLEAR_VETO",
                        "skip_reason": "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3",
                        "sxx": None,
                        "sxx_float_hex": None,
                        "constant_predictor_branch": None,
                        "slope": None,
                        "slope_float_hex": None,
                        "intercept": None,
                        "intercept_float_hex": None,
                        "fit_completed": False,
                    }
                )
    return {
        "schema_version": 3,
        "artifact_type": "TARGET_FREE_FIT_LEDGER_V3",
        "status": amendment["output_contract"]["decision_status_literals_exact"][
            "positive"
        ],
        "planned_counts": dict(seal.FIT_LEDGER_PLANNED_COUNTS),
        "executed_counts": {
            "completed_primary_model_units": 0,
            "completed_analytic_affine_units": 0,
            "completed_total_decision_units": 0,
        },
        "internal_method_call_counts": {
            "sklearn.pipeline.Pipeline.fit": 0,
            "sklearn.preprocessing.StandardScaler.fit": 0,
            "sklearn.linear_model.Ridge.fit": 0,
            "sklearn.ensemble.ExtraTreesRegressor.fit": 0,
            "total_method_calls": 0,
        },
        "skipped_counts": {
            "skipped_primary_model_units": 72,
            "skipped_analytic_affine_units": 1_260,
            "skipped_total_decision_units": 1_332,
        },
        "zero_fit_counts": {"derived": 0, "group": 0, "hyperparameter_or_seed_search": 0},
        "unit_slots": slots,
    }


def _completed_primary_fit_ledger_payload(
    amendment: Mapping[str, Any]
) -> dict[str, Any]:
    ledger = _fit_ledger_payload(amendment)
    for slot in ledger["unit_slots"][:72]:
        ridge = slot["estimator_unit"] == "RIDGE_PIPELINE"
        slot.update(
            {
                "unit_status": "COMPLETED",
                "skip_reason": None,
                "pipeline_fit_calls": int(ridge),
                "standard_scaler_fit_calls": int(ridge),
                "ridge_fit_calls": int(ridge),
                "extra_trees_fit_calls": int(not ridge),
                "predict_calls": 1,
                "fit_completed": True,
            }
        )
    ledger["executed_counts"] = {
        "completed_primary_model_units": 72,
        "completed_analytic_affine_units": 0,
        "completed_total_decision_units": 72,
    }
    ledger["internal_method_call_counts"] = {
        "sklearn.pipeline.Pipeline.fit": 36,
        "sklearn.preprocessing.StandardScaler.fit": 36,
        "sklearn.linear_model.Ridge.fit": 36,
        "sklearn.ensemble.ExtraTreesRegressor.fit": 36,
        "total_method_calls": 144,
    }
    ledger["skipped_counts"] = {
        "skipped_primary_model_units": 0,
        "skipped_analytic_affine_units": 1_260,
        "skipped_total_decision_units": 1_260,
    }
    return ledger


def _selected_response_skipped_fit_ledger_payload(
    amendment: Mapping[str, Any], response: str, estimator: str
) -> dict[str, Any]:
    ledger = _completed_primary_fit_ledger_payload(amendment)
    selected = [
        slot
        for slot in ledger["unit_slots"][:72]
        if slot["response"] == response and slot["estimator_unit"] == estimator
    ]
    assert len(selected) == 4
    for slot in selected:
        slot.update(
            {
                "unit_status": "SKIPPED_PREMODEL_CLEAR_VETO",
                "skip_reason": (
                    "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3"
                ),
                "pipeline_fit_calls": 0,
                "standard_scaler_fit_calls": 0,
                "ridge_fit_calls": 0,
                "extra_trees_fit_calls": 0,
                "predict_calls": 0,
                "fit_completed": False,
            }
        )
    ridge_skipped = estimator == "RIDGE_PIPELINE"
    ledger["executed_counts"] = {
        "completed_primary_model_units": 68,
        "completed_analytic_affine_units": 0,
        "completed_total_decision_units": 68,
    }
    ledger["internal_method_call_counts"] = {
        "sklearn.pipeline.Pipeline.fit": 32 if ridge_skipped else 36,
        "sklearn.preprocessing.StandardScaler.fit": 32 if ridge_skipped else 36,
        "sklearn.linear_model.Ridge.fit": 32 if ridge_skipped else 36,
        "sklearn.ensemble.ExtraTreesRegressor.fit": 36 if ridge_skipped else 32,
        "total_method_calls": 132 if ridge_skipped else 140,
    }
    ledger["skipped_counts"] = {
        "skipped_primary_model_units": 4,
        "skipped_analytic_affine_units": 1_260,
        "skipped_total_decision_units": 1_264,
    }
    return ledger


def _final_stage_fixture(
    artifact_root: Path,
    amendment: Mapping[str, Any],
    status_key: str,
) -> dict[str, dict[str, Any]]:
    output = artifact_root / seal.OUTPUT_ROOT_RELATIVE
    transaction = artifact_root / seal.TRANSACTION_ROOT_RELATIVE
    transaction.mkdir(parents=True)
    for index, alias in enumerate(seal.ALIAS_ORDER, start=1):
        (output / alias).write_bytes(f"alias-{index}\n".encode("ascii"))
    payloads: dict[str, bytes] = {
        "TARGET_FREE_FAMILY_DECISION.json": seal.pretty_json_bytes(
            _decision_payload(amendment, status_key)
        ),
        "TARGET_FREE_FAMILY_LOCK.json": seal.pretty_json_bytes(
            _lock_payload(amendment, status_key)
        ),
        "TRACK_A_TARGET_FREE_RUN_MANIFEST.json": seal.pretty_json_bytes(
            _manifest_payload()
        ),
        "TARGET_FREE_FIT_LEDGER_V3.json": b"{}\n",
        "TARGET_FREE_DISTRIBUTION_STABILITY_V3.json": b"{}\n",
    }
    result: dict[str, dict[str, Any]] = {}
    for ordinal, name in enumerate(seal.FINAL_PUBLICATION_ORDER, start=1):
        relative = seal.RUNNER_STAGED_RELATIVES.get(
            name, seal.SEALER_STAGED_RELATIVES.get(name)
        )
        assert relative is not None
        stage = artifact_root / relative
        data = payloads.get(name, f"fixture-{ordinal}\n".encode("ascii"))
        stage.write_bytes(data)
        result[name] = seal.file_identity(stage, relative_to=artifact_root)
    return result


def _parent_plan_fixture() -> dict[str, Any]:
    return {
        "decode_contract": {
            "spatial_method": (
                "deterministic bilinear interpolation on regular_ll grid, "
                "then capacity-weighted group mean"
            )
        }
    }


def _code_identity_fixture() -> dict[str, dict[str, Any]]:
    return {
        role: _identity(f"C:/fixture/{relative}", marker=chr(97 + ordinal))
        for ordinal, (role, relative) in enumerate(seal.CODE_ROLE_PATHS.items())
    }


def _control_identity_fixture() -> dict[str, dict[str, Any]]:
    records = {
        "code_seal": _identity(seal.CODE_SEAL_RELATIVE, marker="a"),
        "authorization": _identity(seal.AUTHORIZATION_RELATIVE, marker="b"),
        "independent_review": _identity(seal.REVIEW_RELATIVE, marker="c"),
        "independent_go": _identity(seal.GO_RELATIVE, marker="d"),
        "postrun_pass": _identity(seal.POSTRUN_RELATIVE, marker="e"),
    }
    # Deliberately reverse insertion order: JSON objects are maps, never order authority.
    return dict(reversed(tuple(records.items())))


def _finalizer_validation_fixture(
    artifact_root: Path, amendment: Mapping[str, Any]
) -> dict[str, Any]:
    output = artifact_root / seal.OUTPUT_ROOT_RELATIVE
    transaction = artifact_root / seal.TRANSACTION_ROOT_RELATIVE
    transaction.mkdir(parents=True)
    aliases: dict[str, dict[str, Any]] = {}
    for ordinal, name in enumerate(seal.ALIAS_ORDER, start=1):
        path = output / name
        path.write_bytes(f"alias-fixture-{ordinal}\n".encode("ascii"))
        aliases[name] = seal.file_identity(path, relative_to=artifact_root)

    decision = _decision_payload(amendment, "positive")
    fit_ledger = _completed_primary_fit_ledger_payload(amendment)
    json_payloads: dict[str, Any] = {
        "TARGET_FREE_FAMILY_DECISION.json": decision,
        "TARGET_FREE_FIT_LEDGER_V3.json": fit_ledger,
        "TARGET_FREE_DISTRIBUTION_STABILITY_V3.json": {
            "fixture": "publication bytes only"
        },
    }
    runner: dict[str, dict[str, Any]] = {}
    for ordinal, name in enumerate(seal.RUNNER_PUBLICATION_ORDER, start=1):
        path = artifact_root / seal.RUNNER_STAGED_RELATIVES[name]
        data = (
            seal.pretty_json_bytes(json_payloads[name])
            if name in json_payloads
            else f"runner-stage-{ordinal}\n".encode("ascii")
        )
        path.write_bytes(data)
        identity = seal.file_identity(path, relative_to=artifact_root)
        physical_sha = identity["sha256"]
        runner[name] = {
            **identity,
            "format": seal.EXPECTED_RUNNER_FORMATS[name],
            "row_count": seal.EXPECTED_RUNNER_ROWS[name],
            "logical_sha256": (
                physical_sha
                if seal.EXPECTED_RUNNER_FORMATS[name] != "PARQUET"
                else f"{ordinal + 32:064x}"
            ),
        }
    staged = {"aliases": aliases, "runner_staged_outputs": runner}
    controls = {
        "code_seal": {},
        "authorization": {},
        "review": {},
        "go": {},
        "postrun": {"created_utc": "2026-08-11T16:00:08.000000Z"},
    }
    return {
        "amendment": amendment,
        "controls": controls,
        "control_identities": _control_identity_fixture(),
        "attempt_id": "target_free_duplicate_v3__20260811T000000000000Z",
        "code_identities": _code_identity_fixture(),
        "staged_output_identities": staged,
        "staged_decision": decision,
        "staged_fit_ledger": fit_ledger,
        "runner_stdout_payload": {
            "no_forbidden_access_attestation": dict(
                seal.DECISION_FORBIDDEN_ACCESS_ATTESTATION
            )
        },
        "threadpool_info_before_inside_after": (
            seal._expected_threadpool_evidence_from_ledger(fit_ledger, amendment)
        ),
        "models_fit_by_sealer": 0,
        "parquet_logical_value_reads_by_sealer": 0,
    }


def test_normative_amendment_f9d0_ae5c_and_exact_orders(amendment: Mapping[str, Any]) -> None:
    assert seal.AMENDMENT_SIZE_BYTES == 102_361
    assert seal.AMENDMENT_SHA256.startswith("f9d072fc")
    assert seal.AMENDMENT_CANONICAL_SIZE_BYTES == 84_601
    assert seal.AMENDMENT_CANONICAL_SHA256.startswith("ae5c5d05")
    output = amendment["output_contract"]
    complete = [f"{seal.OUTPUT_ROOT_RELATIVE}{name}" for name in seal.EXACT11_SCHEMA_ORDER]
    assert output["positive_complete_paths_exact"] == complete
    assert output["completed_negative_complete_paths_exact"] == complete
    assert output["final_publication_order_positive_exact"] == list(seal.FINAL_PUBLICATION_ORDER)
    assert output["final_publication_order_negative_exact"] == list(seal.FINAL_PUBLICATION_ORDER)
    assert seal.FINAL_PUBLICATION_ORDER[-1] == "TRACK_A_TARGET_FREE_RUN_MANIFEST.json"


def test_normative_v4_exact2_overlay_identity_signature_and_control_delta(
    amendment: Mapping[str, Any],
) -> None:
    root = seal.ARTIFACT_ROOT_DEFAULT
    v3 = seal._validate_v3_amendment(root)
    v4 = seal.validate_amendment_v4(root, v3)
    observed = seal.file_identity(
        root / seal.AMENDMENT_V4_RELATIVE, relative_to=root
    )
    assert observed == {
        key: value
        for key, value in seal.amendment_v4_identity().items()
        if key in seal.FILE_IDENTITY_KEYS
    }
    assert seal.AMENDMENT_V4_SIZE_BYTES == 14_385
    assert seal.AMENDMENT_V4_SHA256.startswith("9f8abbe8")
    assert seal.AMENDMENT_V4_CANONICAL_SIZE_BYTES == 12_501
    assert seal.AMENDMENT_V4_CANONICAL_SHA256.startswith("3ad1f385")
    assert len(v4) == 18
    assert v4["correction_scope"]["superseded_pointers_exact_order"] == list(
        seal.V4_CORRECTION_PATHS
    )
    assert v4["corrected_parquet_serialization"][
        "explicit_write_table_kwargs_exact"
    ] == seal.V4_EXPLICIT_WRITE_TABLE_KWARGS
    seal._validate_v4_pyarrow_signature(v4)
    assert amendment["output_contract"]["output_serialization_exact"][
        "parquet"
    ] == seal.V4_PARQUET_LITERAL
    assert amendment["output_contract"]["output_serialization_exact"][
        "csv_boundary_flags"
    ] == seal.V4_CSV_BOUNDARY_LITERAL
    restored = copy.deepcopy(amendment)
    restored_serialization = restored["output_contract"][
        "output_serialization_exact"
    ]
    restored_serialization["parquet"] = v3["output_contract"][
        "output_serialization_exact"
    ]["parquet"]
    restored_serialization["csv_boundary_flags"] = v3["output_contract"][
        "output_serialization_exact"
    ]["csv_boundary_flags"]
    assert restored == v3
    assert {
        kind: len(spec["keys"]) for kind, spec in seal.CONTROL_SPECS.items()
    } == {
        "code_seal": 12,
        "authorization": 16,
        "review": 15,
        "go": 16,
        "postrun": 17,
    }
    assert len(seal.AUDITOR_STDOUT_KEYS) == 17
    assert len(seal.RUNNER_STDOUT_KEYS) == 11


def test_sealer_runtime_identity_is_exact_and_rejects_executable_drift(
    amendment: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = seal.validate_sealer_runtime_identity(amendment)
    assert observed["python_full_version"] == amendment["runtime_identity"][
        "python_full_version"
    ]
    monkeypatch.setattr(seal.sys, "executable", "C:/unfrozen/python.exe")
    with pytest.raises(seal.TargetFreeSealError, match="executable"):
        seal.validate_sealer_runtime_identity(amendment)


@pytest.mark.parametrize("status_key", ["positive", "clear_no_family", "global_ambiguity"])
def test_decision_nested_exact_closure_for_all_three_terminals(
    amendment: Mapping[str, Any], status_key: str
) -> None:
    decision = _decision_payload(amendment, status_key)
    assert seal.validate_decision_payload(decision, amendment) == decision
    assert len(decision["raw_component_results"]) == 9
    assert len(decision["endpoint_veto_results"]) == 3
    assert "attempt_id" not in decision


def test_decision_nested_rejects_schema_binding_family_priority_and_reason_drift(
    amendment: Mapping[str, Any]
) -> None:
    decision = _decision_payload(amendment, "positive")
    bad = copy.deepcopy(decision)
    bad["raw_component_results"][0]["extra"] = False
    with pytest.raises(seal.TargetFreeSealError, match="raw decision schema"):
        seal.validate_decision_payload(bad, amendment)
    bad = copy.deepcopy(decision)
    endpoint = bad["endpoint_veto_results"][0]
    endpoint["source_component_estimator_binding"]["UGRD_925mb"] = "EXTRA_TREES"
    with pytest.raises(seal.TargetFreeSealError, match="source estimator"):
        seal.validate_decision_payload(bad, amendment)
    bad = copy.deepcopy(decision)
    bad["family_clear_pass_results"]["LOW_LEVEL_ISOBARIC_WIND_PROFILE"] = False
    with pytest.raises(seal.TargetFreeSealError, match="recomputation"):
        seal.validate_decision_payload(bad, amendment)
    bad = copy.deepcopy(decision)
    bad["selected_family"] = "PBL_HEIGHT"
    with pytest.raises(seal.TargetFreeSealError, match="priority"):
        seal.validate_decision_payload(bad, amendment)

    ambiguous = _decision_payload(amendment, "global_ambiguity")
    ambiguous["global_ambiguity_reasons"] = ["AMBIGUOUS_GRAY_ZONE"]
    with pytest.raises(seal.TargetFreeSealError, match="reason"):
        seal.validate_decision_payload(ambiguous, amendment)


def test_threshold_witness_uses_explicit_nonlexical_scope_order_and_exact_hex(
    amendment: Mapping[str, Any]
) -> None:
    decision = _decision_payload(amendment, "positive")
    response = amendment["planned_fit_budget"]["response_order"][0]
    predictors = amendment["input_and_join_contract"]["predictor_columns_exact"]

    def witness(predictor: str, value: float) -> dict[str, Any]:
        return {
            "cutoff_name": "AFFINE_ABS_PEARSON_0.9999",
            "scope": f"RAW/{response}/AFFINE/{predictor}",
            "diagnostic": response,
            "value": value,
            "value_float_hex": value.hex(),
            "cutoff_float_hex": float(0.9999).hex(),
            "boundary_equal": False,
        }

    decision["all_threshold_float_hex_witnesses"] = [
        witness(predictors[0], 0.5),
        witness(predictors[1], 0.6),
    ]
    seal.validate_decision_payload(decision, amendment)
    bad = copy.deepcopy(decision)
    bad["all_threshold_float_hex_witnesses"].reverse()
    with pytest.raises(seal.TargetFreeSealError, match="explicit frozen order"):
        seal.validate_decision_payload(bad, amendment)
    bad = copy.deepcopy(decision)
    bad["all_threshold_float_hex_witnesses"][0]["scope"] = (
        f"RAW/{response}/AFFINE/not_a_frozen_predictor"
    )
    with pytest.raises(seal.TargetFreeSealError, match="scope"):
        seal.validate_decision_payload(bad, amendment)
    bad = copy.deepcopy(decision)
    bad["all_threshold_float_hex_witnesses"][0]["value_float_hex"] = 0.75.hex()
    with pytest.raises(seal.TargetFreeSealError, match="float-hex"):
        seal.validate_decision_payload(bad, amendment)
    bad = copy.deepcopy(decision)
    bad["all_threshold_float_hex_witnesses"][0]["boundary_equal"] = True
    with pytest.raises(seal.TargetFreeSealError, match="boundary equality"):
        seal.validate_decision_payload(bad, amendment)


def test_parent_tie_can_be_masked_only_by_prior_affine_redundancy(
    amendment: Mapping[str, Any]
) -> None:
    decision = _decision_payload(amendment, "clear_no_family")
    raw = decision["raw_component_results"][0]
    raw["parent_reference_estimators"] = list(seal.ESTIMATOR_ENUM)
    raw["parent_redundancy_boolean"] = None
    decision["all_tied_witnesses"] = [
        {
            "tie_kind": "PARENT_MAX_R2",
            "diagnostic": raw["response"],
            "estimators": list(seal.ESTIMATOR_ENUM),
            "metric_name": "R2",
            "metric_values_float_hex": [0.5.hex(), 0.5.hex()],
            "resolution": "MASKED_BY_PRIOR_AFFINE_REDUNDANCY",
            "global_ambiguity": False,
        }
    ]
    seal.validate_decision_payload(decision, amendment)
    bad = copy.deepcopy(decision)
    bad["all_tied_witnesses"][0]["resolution"] = "GLOBAL_AMBIGUITY"
    bad["all_tied_witnesses"][0]["global_ambiguity"] = True
    with pytest.raises(seal.TargetFreeSealError, match="parent global tie"):
        seal.validate_decision_payload(bad, amendment)
    bad = copy.deepcopy(decision)
    bad["all_tied_witnesses"][0]["metric_values_float_hex"][1] = 0.75.hex()
    with pytest.raises(seal.TargetFreeSealError, match="bit-exactly equal"):
        seal.validate_decision_payload(bad, amendment)
    bad = copy.deepcopy(decision)
    bad["all_tied_witnesses"] = []
    with pytest.raises(seal.TargetFreeSealError, match="cardinality"):
        seal.validate_decision_payload(bad, amendment)
    bad = copy.deepcopy(decision)
    bad["raw_component_results"][0]["parent_reference_estimators"] = [
        "RIDGE_PIPELINE"
    ]
    bad["all_tied_witnesses"] = []
    with pytest.raises(seal.TargetFreeSealError, match="two-estimator"):
        seal.validate_decision_payload(bad, amendment)


def test_endpoint_raw_tie_precedes_a_coexisting_premodel_clear(
    amendment: Mapping[str, Any]
) -> None:
    decision = _decision_payload(amendment, "positive")
    raw_by_response = {
        record["response"]: record
        for record in decision["raw_component_results"]
    }
    for response, classification in (
        ("UGRD_925mb", "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"),
        ("UGRD_1000mb", "CLEAR_PHYSICAL_OR_COVERAGE_FAILURE"),
    ):
        raw_by_response[response].update(
            {
                "classification": classification,
                "counts_as_nonredundant": False,
                "independent_estimator": None,
                "parent_reference_estimators": [],
                "parent_redundancy_boolean": None,
                "affine_qualifying_predictors": [],
                "cell_stability_pass": None,
                "distribution_stability_pass": None,
                "boundary_equality_flags": [],
                "ambiguity_reason": (
                    classification if classification.startswith("AMBIGUOUS_") else None
                ),
            }
        )
    for endpoint in decision["endpoint_veto_results"]:
        sources = seal.ENDPOINT_SOURCE_RESPONSES[endpoint["diagnostic"]]
        endpoint["source_component_estimator_binding"] = {
            response: raw_by_response[response]["independent_estimator"]
            for response in sources
        }
        if "UGRD_925mb" in sources:
            endpoint.update(
                {
                    "verdict": "AMBIGUOUS_RAW_ESTIMATOR_TIE",
                    "pooled_stable_margin_pass": None,
                    "cell_stability_pass": None,
                    "distribution_stability_pass": None,
                    "boundary_equality_flags": [],
                    "clear_pass": False,
                    "ambiguity_reason": "AMBIGUOUS_RAW_ESTIMATOR_TIE",
                }
            )
    decision["all_tied_witnesses"] = [
        {
            "tie_kind": "INDEPENDENT_RMSE",
            "diagnostic": "UGRD_925mb",
            "estimators": list(seal.ESTIMATOR_ENUM),
            "metric_name": "RMSE",
            "metric_values_float_hex": [1.0.hex(), 1.0.hex()],
            "resolution": "GLOBAL_AMBIGUITY",
            "global_ambiguity": True,
        }
    ]
    decision["global_ambiguity"] = True
    decision["global_ambiguity_reasons"] = [
        "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE:UGRD_925mb"
    ]
    decision["family_clear_pass_results"]["LOW_LEVEL_ISOBARIC_WIND_PROFILE"] = False
    decision["status"] = amendment["output_contract"][
        "decision_status_literals_exact"
    ]["global_ambiguity"]
    decision["selected_family"] = None
    decision["selected_raw_columns"] = []
    decision["selected_derived_columns"] = []
    seal.validate_decision_payload(decision, amendment)


def test_wind_salvages_four_constant_middle_level_components_but_not_physical_failure(
    amendment: Mapping[str, Any]
) -> None:
    decision = _decision_payload(amendment, "positive")
    constant_names = {
        "UGRD_950mb", "VGRD_950mb", "UGRD_975mb", "VGRD_975mb"
    }
    for raw in decision["raw_component_results"]:
        if raw["response"] not in constant_names:
            continue
        raw.update(
            {
                "classification": "CLEAR_NONNOVEL_CONSTANT",
                "counts_as_nonredundant": False,
                "independent_estimator": None,
                "parent_reference_estimators": [],
                "parent_redundancy_boolean": None,
                "affine_qualifying_predictors": [],
                "cell_stability_pass": None,
                "distribution_stability_pass": None,
                "boundary_equality_flags": [],
                "ambiguity_reason": None,
            }
        )
    seal.validate_decision_payload(decision, amendment)
    bad = copy.deepcopy(decision)
    physical = next(
        row for row in bad["raw_component_results"] if row["response"] == "UGRD_950mb"
    )
    physical["classification"] = "CLEAR_PHYSICAL_OR_COVERAGE_FAILURE"
    with pytest.raises(seal.TargetFreeSealError, match="family clear-pass"):
        seal.validate_decision_payload(bad, amendment)


@pytest.mark.parametrize(
    "raw",
    [
        '{"a":1,"a":2}',
        '{"a":{"b":1,"b":2}}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        '{"x":1e400}',
    ],
)
def test_strict_json_rejects_duplicates_and_every_nonfinite_route(raw: str) -> None:
    with pytest.raises(seal.TargetFreeSealError):
        seal.strict_json_loads(raw, label="fixture")


def test_json_serializers_are_deterministic_ascii_lf_and_finite() -> None:
    payload = {"z": "한글", "a": [1, 2.5, None]}
    pretty = seal.pretty_json_bytes(payload)
    compact = seal.compact_json_bytes(payload)
    assert pretty.endswith(b"\n") and not pretty.startswith(b"\xef\xbb\xbf")
    assert b"\\ud55c\\uae00" in pretty
    assert compact == b'{"a":[1,2.5,null],"z":"\\ud55c\\uae00"}'
    with pytest.raises(seal.TargetFreeSealError):
        seal.pretty_json_bytes({"x": float("nan")})


def test_json_file_loader_requires_exact_pretty_ascii_serialization(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    payload = {"z": 2, "a": 1}
    path.write_bytes(seal.pretty_json_bytes(payload))
    assert seal.strict_json_load(path, label="control") == payload
    path.write_bytes(b'{"a":1,"z":2}\n')
    with pytest.raises(seal.TargetFreeSealError, match="serialization"):
        seal.strict_json_load(path, label="control")


def test_identity_and_path_guards_are_exact(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "record.bin"
    path.write_bytes(b"bound-bytes")
    record = seal.file_identity(path, relative_to=root)
    assert seal.validate_identity_record(
        record, path, label="record", relative_to=root
    ) == record
    wrong = dict(record, size_bytes=True)
    with pytest.raises(seal.TargetFreeSealError):
        seal.validate_identity_record(
            wrong, path, label="record", relative_to=root, rehash=False
        )
    with pytest.raises(seal.TargetFreeSealError):
        seal.path_under(root, "../escape", label="escape")
    with pytest.raises(seal.TargetFreeSealError):
        seal.path_under(root, "back\\slash", label="backslash")


def test_attempt_id_has_strict_prefix_shape_and_calendar_valid_microseconds() -> None:
    valid = "target_free_duplicate_v3__20260811T235959123456Z"
    assert seal.require_attempt_id(valid, label="fixture") == valid
    for invalid in (
        "target_free_duplicate_v3__fixture",
        "target_free_duplicate_v3__20260230T000000000000Z",
        "target_free_duplicate_v3__20260811T235960000000Z",
        "other__20260811T235959123456Z",
    ):
        with pytest.raises(seal.TargetFreeSealError, match="attempt_id"):
            seal.require_attempt_id(invalid, label="fixture")


def test_single_hardlink_publisher_is_create_if_absent_and_rehashed(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    final = tmp_path / "final"
    data = b"append-only-publication\n"
    seal.stage_bytes_exclusive(stage, data)
    result = seal.publish_staged_create_if_absent(
        stage,
        final,
        expected_size=len(data),
        expected_sha256=_sha(data),
    )
    assert not stage.exists()
    assert final.read_bytes() == data
    assert result["sha256"] == _sha(data)
    if hasattr(final.stat(), "st_nlink"):
        assert final.stat().st_nlink == 1

    second_stage = tmp_path / "stage-second"
    seal.stage_bytes_exclusive(second_stage, data)
    with pytest.raises(seal.TargetFreeSealError, match="already exists"):
        seal.publish_staged_create_if_absent(
            second_stage,
            final,
            expected_size=len(data),
            expected_sha256=_sha(data),
        )
    assert final.read_bytes() == data
    assert second_stage.exists()


def _tiny_alias_specs(root: Path) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(seal.ALIAS_ORDER, start=1):
        source_relative = f"source-{index}.bin"
        data = f"source-{index}-independent-bytes\n".encode("ascii")
        (root / source_relative).write_bytes(data)
        specs[name] = {
            "source": source_relative,
            "stage": f"{seal.OUTPUT_ROOT_RELATIVE}.stage-{index}",
            "final": f"{seal.OUTPUT_ROOT_RELATIVE}{name}",
            "size_bytes": len(data),
            "sha256": _sha(data),
            "bundle": (),
        }
    return specs


def test_alias_pair_copies_both_before_publish_and_severs_source_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    specs = _tiny_alias_specs(root)
    events: list[str] = []
    real_copy = seal.stage_copy_exclusive
    real_publish = seal.publish_staged_create_if_absent

    def copy_spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append(f"copy:{Path(args[1]).name}")
        return real_copy(*args, **kwargs)

    def publish_spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append(f"publish:{Path(args[1]).name}")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(seal, "stage_copy_exclusive", copy_spy)
    monkeypatch.setattr(seal, "publish_staged_create_if_absent", publish_spy)
    observed = seal.publish_alias_pair_predata(root, specs)
    assert events[:2] == ["copy:.stage-1", "copy:.stage-2"]
    assert events[2:] == [
        "publish:FIELD_CENSUS_LOCK.json",
        "publish:PROVENANCE_LEDGER.parquet",
    ]
    for name, spec in specs.items():
        source = root / spec["source"]
        final = root / spec["final"]
        assert final.read_bytes() == source.read_bytes()
        assert not os.path.samefile(source, final)
        assert observed[name]["path"] == spec["final"]


def test_final_alias_semantic_and_footer_closure_is_metadata_only_and_tamper_strict(
    tmp_path: Path,
    amendment: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "artifact"
    output = root / seal.OUTPUT_ROOT_RELATIVE
    output.mkdir(parents=True)

    def referenced_identity(relative: str, marker: str) -> dict[str, Any]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((marker + "\n").encode("ascii"))
        return seal.file_identity(path, relative_to=root)

    identity_fields = {
        field: referenced_identity(f"census/fixture_{index:02d}.dat", field)
        for index, field in enumerate(
            seal.CENSUS_ALIAS_IDENTITY_FIELDS, start=1
        )
    }
    checkpoints = [
        referenced_identity(
            f"census/progress/fixture_{index:02d}.json", str(index)
        )
        for index in range(13)
    ]
    field_payload = {
        **identity_fields,
        "artifact_type": "NOAA_GFS_MULTISEASON_CENSUS_MANIFEST",
        "created_utc": "2026-08-10T13:47:10.975928Z",
        "labels_read": False,
        "models_fit": 0,
        "progress_checkpoints": checkpoints,
        "raw_downloaded_bytes": 0,
        "raw_network_requests": 0,
        "schema_version": 1,
        "submission_csv_created": False,
    }
    assert set(field_payload) == seal.CENSUS_ALIAS_TOP_KEYS
    field_path = output / "FIELD_CENSUS_LOCK.json"
    field_bytes = seal.pretty_json_bytes(field_payload)
    field_path.write_bytes(field_bytes)
    field_spec = seal.ALIAS_SPECS["FIELD_CENSUS_LOCK.json"]
    monkeypatch.setitem(field_spec, "size_bytes", len(field_bytes))
    monkeypatch.setitem(field_spec, "sha256", _sha(field_bytes))

    frozen = amendment["output_contract"]["upstream_alias_closure"]["aliases"][
        "PROVENANCE_LEDGER.parquet"
    ]
    required = frozen["required_preregister_provenance_fields_exact"]
    rows = 10_368
    provenance_path = output / "PROVENANCE_LEDGER.parquet"

    def write_provenance(columns: list[str], row_count: int) -> None:
        table = pa.table(
            {name: pa.nulls(row_count, type=pa.string()) for name in columns}
        )
        pq.write_table(table, provenance_path)
        provenance_spec = seal.ALIAS_SPECS["PROVENANCE_LEDGER.parquet"]
        monkeypatch.setitem(
            provenance_spec, "size_bytes", provenance_path.stat().st_size
        )
        monkeypatch.setitem(
            provenance_spec, "sha256", seal.sha256_file(provenance_path)
        )

    write_provenance(list(required), rows)
    report = seal.validate_final_alias_schema_closure(root, amendment)
    assert report["provenance_footer_rows"] == rows
    assert report["required_exact15_present"] is True
    assert report["parquet_logical_values_read"] == 0

    bad_field = copy.deepcopy(field_payload)
    bad_field["labels_read"] = True
    bad_bytes = seal.pretty_json_bytes(bad_field)
    field_path.write_bytes(bad_bytes)
    monkeypatch.setitem(field_spec, "size_bytes", len(bad_bytes))
    monkeypatch.setitem(field_spec, "sha256", _sha(bad_bytes))
    with pytest.raises(seal.TargetFreeSealError, match="labels_read"):
        seal.validate_final_alias_schema_closure(root, amendment)

    field_path.write_bytes(field_bytes)
    monkeypatch.setitem(field_spec, "size_bytes", len(field_bytes))
    monkeypatch.setitem(field_spec, "sha256", _sha(field_bytes))
    write_provenance(list(required[:-1]), rows)
    with pytest.raises(seal.TargetFreeSealError, match="required exact15"):
        seal.validate_final_alias_schema_closure(root, amendment)
    write_provenance(list(required), rows - 1)
    with pytest.raises(seal.TargetFreeSealError, match="row count"):
        seal.validate_final_alias_schema_closure(root, amendment)


def test_alias_partial_publication_is_terminal_and_never_cleaned_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    specs = _tiny_alias_specs(root)
    real_publish = seal.publish_staged_create_if_absent
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise seal.TargetFreeSealError("injected second-alias failure")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(seal, "publish_staged_create_if_absent", fail_second)
    with pytest.raises(seal.PartialPublicationError) as caught:
        seal.publish_alias_pair_predata(root, specs)
    assert caught.value.published == (seal.ALIAS_ORDER[0],)
    assert (root / specs[seal.ALIAS_ORDER[0]]["final"]).exists()
    assert (root / specs[seal.ALIAS_ORDER[1]]["stage"]).exists()


@pytest.mark.parametrize("status_key", ["positive", "clear_no_family", "global_ambiguity"])
def test_exact11_positive_negative_and_ambiguity_publish_manifest_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    amendment: Mapping[str, Any],
    status_key: str,
) -> None:
    root = tmp_path / status_key
    root.mkdir()
    staged = _final_stage_fixture(root, amendment, status_key)
    calls: list[str] = []
    real_publish = seal.publish_staged_create_if_absent

    def publish_spy(stage: Path, destination: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append(destination.name)
        return real_publish(stage, destination, **kwargs)

    monkeypatch.setattr(seal, "publish_staged_create_if_absent", publish_spy)
    _outputs, published = seal.publish_final_sequence(root, staged)
    assert published == seal.FINAL_PUBLICATION_ORDER
    assert calls == list(seal.FINAL_PUBLICATION_ORDER)
    assert calls[-1] == "TRACK_A_TARGET_FREE_RUN_MANIFEST.json"
    alias_schema_checks: list[Path] = []

    def alias_schema_fixture(
        artifact_root: Path, _amendment: Mapping[str, Any]
    ) -> dict[str, Any]:
        alias_schema_checks.append(artifact_root)
        return {"parquet_logical_values_read": 0}

    monkeypatch.setattr(
        seal, "validate_final_alias_schema_closure", alias_schema_fixture
    )
    identities = seal.validate_exact11_completion(
        root, amendment=amendment, require_transaction_absent=False
    )
    assert alias_schema_checks == [root]
    assert tuple(identities) == seal.EXACT11_SCHEMA_ORDER
    assert len(identities) == 11


@pytest.mark.parametrize("status_key", ["positive", "clear_no_family", "global_ambiguity"])
def test_family_lock_builder_binds_frozen_nested_contract_and_terminal_nullability(
    amendment: Mapping[str, Any], status_key: str
) -> None:
    decision = _decision_payload(amendment, status_key)
    code = _code_identity_fixture()
    postrun = _identity(seal.POSTRUN_RELATIVE, marker="e")
    lock = seal.build_family_lock_payload(
        amendment=amendment,
        decision=decision,
        code_identities=code,
        postrun_pass_identity=postrun,
        parent_plan=_parent_plan_fixture(),
    )
    assert set(lock) == seal.LOCK_TOP_KEYS
    assert lock["parent_and_independent_gate_results"] == decision
    assert lock["no_label_attestation"] == seal.DECISION_FORBIDDEN_ACCESS_ATTESTATION
    assert tuple(lock["code_config_runtime_identities"]) == (
        "amendment",
        "amendment_v4",
        "code_identities",
        "runtime_identity",
    )
    assert lock["code_config_runtime_identities"]["amendment_v4"] == (
        seal.amendment_v4_identity()
    )
    assert set(lock["site_group_scope"]) == set(seal.SITE_GROUP_SCOPE_KEYS)
    assert lock["spatial_transforms"] == {
        "spatial_method": _parent_plan_fixture()["decode_contract"]["spatial_method"],
        "group_construction": amendment["derived_wind_contract"]["group_construction"],
    }
    if status_key == "positive":
        expected_columns = [
            *decision["selected_raw_columns"],
            *decision["selected_derived_columns"],
        ]
        assert set(lock["units"]) == set(expected_columns)
        assert lock["units"]["HPBL_surface"] == "m" if "HPBL_surface" in lock["units"] else True
        assert lock["units"]["ENDPOINT_DIRECTION_COS"] == "dimensionless"
        assert set(lock["grib_selectors"]) == set(decision["selected_raw_columns"])
    else:
        for field in amendment["output_contract"]["output_record_schemas_exact"][
            "TARGET_FREE_FAMILY_LOCK_json"
        ]["negative_tombstone_null_fields"]:
            assert lock[field] is None
    round_trip = seal.strict_json_loads(
        seal.pretty_json_bytes(lock).decode("utf-8"), label="lock roundtrip"
    )
    assert seal.validate_family_lock_payload(
        round_trip,
        amendment=amendment,
        decision=decision,
        code_identities=code,
        postrun_pass_identity=postrun,
        parent_plan=_parent_plan_fixture(),
    ) == lock
    bad = copy.deepcopy(lock)
    bad["no_label_attestation"] = True
    with pytest.raises(seal.TargetFreeSealError, match="family-lock exact payload"):
        seal.validate_family_lock_payload(
            bad,
            amendment=amendment,
            decision=decision,
            code_identities=code,
            postrun_pass_identity=postrun,
            parent_plan=_parent_plan_fixture(),
        )
    bad = copy.deepcopy(lock)
    bad["code_config_runtime_identities"].pop("amendment_v4")
    with pytest.raises(seal.TargetFreeSealError, match="family-lock exact payload"):
        seal.validate_family_lock_payload(
            bad,
            amendment=amendment,
            decision=decision,
            code_identities=code,
            postrun_pass_identity=postrun,
            parent_plan=_parent_plan_fixture(),
        )


def test_manifest_builder_binds_exact8_fit5_oof4_and_map_order_is_not_authority(
    amendment: Mapping[str, Any],
) -> None:
    decision = _decision_payload(amendment, "positive")
    ledger = _completed_primary_fit_ledger_payload(amendment)
    staged = _staged_identities()
    controls = {
        "code_seal": {},
        "authorization": {},
        "review": {},
        "go": {},
        "postrun": {"created_utc": "2026-08-11T16:00:08.000000Z"},
    }
    control_identities = _control_identity_fixture()
    lock_sha = "f" * 64
    lock_identity = {
        "path": seal.SEALER_STAGED_RELATIVES["TARGET_FREE_FAMILY_LOCK.json"],
        "size_bytes": 100,
        "sha256": lock_sha,
        "format": "JSON",
        "row_count": 1,
        "logical_sha256": lock_sha,
    }
    threadpool = seal._expected_threadpool_evidence_from_ledger(ledger, amendment)
    manifest = seal.build_run_manifest_payload(
        amendment=amendment,
        decision=decision,
        fit_ledger=ledger,
        code_identities=_code_identity_fixture(),
        control_identities=control_identities,
        controls=controls,
        staged_output_identities=staged,
        lock_output_identity=lock_identity,
        threadpool_info=threadpool,
        no_forbidden_access_attestation=seal.DECISION_FORBIDDEN_ACCESS_ATTESTATION,
        created_utc="2026-08-11T16:00:09.000000Z",
    )
    assert set(manifest) == seal.MANIFEST_TOP_KEYS
    assert tuple(manifest["output_identities"]) == seal.MANIFEST_OUTPUT_ORDER
    assert "TRACK_A_TARGET_FREE_RUN_MANIFEST.json" not in manifest["output_identities"]
    assert set(manifest["planned_completed_skipped_fit_counts"]) == set(
        seal.MANIFEST_FIT_COUNT_KEYS
    )
    assert tuple(manifest["code_config_runtime_identities"]) == (
        "amendment",
        "amendment_v4",
        "code_identities",
        "runtime_identity",
    )
    assert manifest["oof_slot_counts"] == {
        "planned_oof_slots": 352_512,
        "completed_oof_slots": 352_512,
        "skipped_oof_slots": 0,
        "independent_selected_oof_slots": 176_256,
    }
    reordered = copy.deepcopy(manifest)
    reordered["control_identities"] = dict(
        reversed(tuple(reordered["control_identities"].items()))
    )
    serialized = seal.pretty_json_bytes(reordered)
    round_trip = seal.strict_json_loads(
        serialized.decode("utf-8"), label="manifest roundtrip"
    )
    assert seal.validate_run_manifest_payload(
        round_trip,
        amendment=amendment,
        decision=decision,
        fit_ledger=ledger,
        code_identities=_code_identity_fixture(),
        control_identities=control_identities,
        controls=controls,
        staged_output_identities=staged,
        lock_output_identity=lock_identity,
        threadpool_info=threadpool,
        no_forbidden_access_attestation=seal.DECISION_FORBIDDEN_ACCESS_ATTESTATION,
    ) == round_trip
    bad = copy.deepcopy(manifest)
    bad["oof_slot_counts"]["completed_oof_slots"] = 1
    with pytest.raises(seal.TargetFreeSealError, match="run-manifest exact payload"):
        seal.validate_run_manifest_payload(
            bad,
            amendment=amendment,
            decision=decision,
            fit_ledger=ledger,
            code_identities=_code_identity_fixture(),
            control_identities=control_identities,
            controls=controls,
            staged_output_identities=staged,
            lock_output_identity=lock_identity,
            threadpool_info=threadpool,
            no_forbidden_access_attestation=seal.DECISION_FORBIDDEN_ACCESS_ATTESTATION,
        )
    bad = copy.deepcopy(manifest)
    bad["code_config_runtime_identities"].pop("amendment_v4")
    with pytest.raises(seal.TargetFreeSealError, match="run-manifest exact payload"):
        seal.validate_run_manifest_payload(
            bad,
            amendment=amendment,
            decision=decision,
            fit_ledger=ledger,
            code_identities=_code_identity_fixture(),
            control_identities=control_identities,
            controls=controls,
            staged_output_identities=staged,
            lock_output_identity=lock_identity,
            threadpool_info=threadpool,
            no_forbidden_access_attestation=seal.DECISION_FORBIDDEN_ACCESS_ATTESTATION,
        )
    all_skipped = _fit_ledger_payload(amendment)
    with pytest.raises(seal.TargetFreeSealError, match="OOF slot accounting"):
        seal.build_run_manifest_payload(
            amendment=amendment,
            decision=decision,
            fit_ledger=all_skipped,
            code_identities=_code_identity_fixture(),
            control_identities=control_identities,
            controls=controls,
            staged_output_identities=staged,
            lock_output_identity=lock_identity,
            threadpool_info=_zero_fit_threadpool(amendment),
            no_forbidden_access_attestation=seal.DECISION_FORBIDDEN_ACCESS_ATTESTATION,
            created_utc="2026-08-11T16:00:09.000000Z",
        )
    first_selected = decision["raw_component_results"][0]
    substituted = _selected_response_skipped_fit_ledger_payload(
        amendment,
        first_selected["response"],
        first_selected["independent_estimator"],
    )
    seal.validate_fit_ledger_payload(
        substituted, amendment, expected_status=decision["status"]
    )
    assert (
        len(
            [
                slot
                for slot in substituted["unit_slots"][:72]
                if slot["fit_completed"] is True
            ]
        )
        * 4_896
        >= 176_256
    )
    with pytest.raises(
        seal.TargetFreeSealError, match="selected OOF ledger crosslink"
    ):
        seal.build_run_manifest_payload(
            amendment=amendment,
            decision=decision,
            fit_ledger=substituted,
            code_identities=_code_identity_fixture(),
            control_identities=control_identities,
            controls=controls,
            staged_output_identities=staged,
            lock_output_identity=lock_identity,
            threadpool_info=seal._expected_threadpool_evidence_from_ledger(
                substituted, amendment
            ),
            no_forbidden_access_attestation=seal.DECISION_FORBIDDEN_ACCESS_ATTESTATION,
            created_utc="2026-08-11T16:00:09.000000Z",
        )


def test_finalizer_tmp_exact11_manifest_last_and_transaction_removed_only_after_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    amendment: Mapping[str, Any],
) -> None:
    root = tmp_path / "finalize-success"
    root.mkdir()
    validation = _finalizer_validation_fixture(root, amendment)
    monkeypatch.setattr(
        seal, "load_bound_parent_plan", lambda *_args, **_kwargs: _parent_plan_fixture()
    )
    monkeypatch.setattr(
        seal,
        "validate_staged_output_identities",
        lambda value, **_kwargs: dict(value),
    )
    alias_checks: list[Path] = []

    def alias_fixture(
        artifact_root: Path, _amendment: Mapping[str, Any]
    ) -> dict[str, Any]:
        alias_checks.append(artifact_root)
        return {"parquet_logical_values_read": 0}

    monkeypatch.setattr(seal, "validate_final_alias_schema_closure", alias_fixture)
    calls: list[str] = []
    real_publish = seal.publish_staged_create_if_absent

    def publish_spy(stage: Path, destination: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append(destination.name)
        return real_publish(stage, destination, **kwargs)

    monkeypatch.setattr(seal, "publish_staged_create_if_absent", publish_spy)
    report = seal.finalize_validated_authority_chain(
        root,
        validation,
        created_utc="2026-08-11T16:00:09.000000Z",
    )
    output = root / seal.OUTPUT_ROOT_RELATIVE
    transaction = root / seal.TRANSACTION_ROOT_RELATIVE
    assert report["status"] == "PASS_V3_EXACT11_COMMITTED_MANIFEST_LAST"
    assert tuple(report["published_order"]) == seal.FINAL_PUBLICATION_ORDER
    assert calls == list(seal.FINAL_PUBLICATION_ORDER)
    assert calls[-1] == "TRACK_A_TARGET_FREE_RUN_MANIFEST.json"
    assert not transaction.exists()
    assert {path.name for path in output.iterdir()} == set(seal.EXACT11_BASENAMES)
    assert alias_checks == [root, root]
    manifest = seal.strict_json_load(
        output / "TRACK_A_TARGET_FREE_RUN_MANIFEST.json", label="final manifest"
    )
    assert manifest["manifest_is_last_commit_marker"] is True
    assert set(manifest["output_identities"]) == set(seal.MANIFEST_OUTPUT_ORDER)


def test_manifest_oof_null_selectors_contribute_zero_selected_slots(
    amendment: Mapping[str, Any],
) -> None:
    decision = _no_selector_decision_payload(amendment)
    seal.validate_decision_payload(decision, amendment)
    ledger = _fit_ledger_payload(amendment)
    ledger["status"] = decision["status"]
    seal.validate_fit_ledger_payload(
        ledger, amendment, expected_status=decision["status"]
    )
    counts = seal._manifest_oof_slot_counts(
        amendment=amendment,
        decision=decision,
        fit_ledger=ledger,
        staged_output_identities=_staged_identities(),
    )
    assert counts == {
        "planned_oof_slots": 352_512,
        "completed_oof_slots": 0,
        "skipped_oof_slots": 352_512,
        "independent_selected_oof_slots": 0,
    }


def test_finalizer_partial_failure_preserves_transaction_and_never_cleans_or_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    amendment: Mapping[str, Any],
) -> None:
    root = tmp_path / "finalize-failure"
    root.mkdir()
    validation = _finalizer_validation_fixture(root, amendment)
    monkeypatch.setattr(
        seal, "load_bound_parent_plan", lambda *_args, **_kwargs: _parent_plan_fixture()
    )
    monkeypatch.setattr(
        seal,
        "validate_staged_output_identities",
        lambda value, **_kwargs: dict(value),
    )
    real_publish = seal.publish_staged_create_if_absent
    calls = 0

    def fail_second(stage: Path, destination: Path, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise seal.TargetFreeSealError("injected finalizer failure")
        return real_publish(stage, destination, **kwargs)

    monkeypatch.setattr(seal, "publish_staged_create_if_absent", fail_second)
    with pytest.raises(seal.PartialPublicationError) as caught:
        seal.finalize_validated_authority_chain(
            root,
            validation,
            created_utc="2026-08-11T16:00:09.000000Z",
        )
    output = root / seal.OUTPUT_ROOT_RELATIVE
    transaction = root / seal.TRANSACTION_ROOT_RELATIVE
    assert caught.value.published == (seal.FINAL_PUBLICATION_ORDER[0],)
    assert transaction.is_dir()
    assert any(transaction.iterdir())
    assert (output / seal.FINAL_PUBLICATION_ORDER[0]).is_file()
    assert not (output / "TRACK_A_TARGET_FREE_RUN_MANIFEST.json").exists()


def test_final_publication_preflights_all_nine_and_does_not_overwrite(
    tmp_path: Path, amendment: Mapping[str, Any]
) -> None:
    root = tmp_path / "preflight"
    root.mkdir()
    staged = _final_stage_fixture(root, amendment, "positive")
    manifest = root / seal.FINAL_RELATIVES["TRACK_A_TARGET_FREE_RUN_MANIFEST.json"]
    manifest.write_bytes(b"preexisting-commit-marker\n")
    with pytest.raises(seal.TargetFreeSealError, match="already exists"):
        seal.publish_final_sequence(root, staged)
    assert manifest.read_bytes() == b"preexisting-commit-marker\n"
    assert not (root / seal.FINAL_RELATIVES[seal.FINAL_PUBLICATION_ORDER[0]]).exists()


def test_final_partial_publication_is_terminal_and_preserved_for_incident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    amendment: Mapping[str, Any],
) -> None:
    root = tmp_path / "partial-final"
    root.mkdir()
    staged = _final_stage_fixture(root, amendment, "positive")
    real_publish = seal.publish_staged_create_if_absent
    calls = 0

    def fail_second(stage: Path, destination: Path, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise seal.TargetFreeSealError("injected final failure")
        return real_publish(stage, destination, **kwargs)

    monkeypatch.setattr(seal, "publish_staged_create_if_absent", fail_second)
    with pytest.raises(seal.PartialPublicationError) as caught:
        seal.publish_final_sequence(root, staged)
    assert caught.value.published == (seal.FINAL_PUBLICATION_ORDER[0],)
    assert (root / seal.FINAL_RELATIVES[seal.FINAL_PUBLICATION_ORDER[0]]).exists()
    assert not (root / seal.FINAL_RELATIVES[seal.FINAL_PUBLICATION_ORDER[1]]).exists()
    assert (root / seal.RUNNER_STAGED_RELATIVES[seal.FINAL_PUBLICATION_ORDER[1]]).exists()


def test_staged_output_identity_exact2_exact7_and_logical_hash_rules() -> None:
    staged = _staged_identities()
    assert seal.validate_staged_output_identities(staged) == staged
    bad = copy.deepcopy(staged)
    bad["runner_staged_outputs"]["TARGET_FREE_DUPLICATE_METRICS.csv"]["logical_sha256"] = "f" * 64
    with pytest.raises(seal.TargetFreeSealError, match="logical/physical"):
        seal.validate_staged_output_identities(bad)
    bad = copy.deepcopy(staged)
    bad["runner_staged_outputs"]["TARGET_FREE_DUPLICATE_OOF_V3.parquet"]["row_count"] += 1
    with pytest.raises(seal.TargetFreeSealError, match="row count"):
        seal.validate_staged_output_identities(bad)


def test_fit_ledger_binds_exact_72_plus_1260_slots_fold_strings_and_zero_refits(
    amendment: Mapping[str, Any],
) -> None:
    ledger = _fit_ledger_payload(amendment)
    expected_status = amendment["output_contract"]["decision_status_literals_exact"][
        "positive"
    ]
    assert seal.validate_fit_ledger_payload(
        ledger, amendment, expected_status=expected_status
    ) == ledger
    assert len(ledger["unit_slots"][:72]) == 72
    assert len(ledger["unit_slots"][72:]) == 1_260
    bad = copy.deepcopy(ledger)
    bad["unit_slots"][0]["fold"] = 0
    with pytest.raises(seal.TargetFreeSealError, match="frozen order"):
        seal.validate_fit_ledger_payload(bad, amendment)
    bad = copy.deepcopy(ledger)
    bad["unit_slots"][0]["pipeline_fit_calls"] = 1
    with pytest.raises(seal.TargetFreeSealError, match="call count"):
        seal.validate_fit_ledger_payload(bad, amendment)
    bad = copy.deepcopy(ledger)
    bad["unit_slots"][0]["decision_fit_ordinal"] = True
    with pytest.raises(seal.TargetFreeSealError, match="decision ordinal"):
        seal.validate_fit_ledger_payload(bad, amendment)
    bad = copy.deepcopy(ledger)
    bad["unit_slots"][72]["sxx"] = 0.0
    bad["unit_slots"][72]["sxx_float_hex"] = 0.0.hex()
    with pytest.raises(seal.TargetFreeSealError, match="skipped affine numeric"):
        seal.validate_fit_ledger_payload(bad, amendment)
    bad = copy.deepcopy(ledger)
    bad["zero_fit_counts"]["group"] = 1
    with pytest.raises(seal.TargetFreeSealError, match="zero-fit"):
        seal.validate_fit_ledger_payload(bad, amendment)
    bad = copy.deepcopy(ledger)
    bad["executed_counts"]["completed_total_decision_units"] = True
    with pytest.raises(seal.TargetFreeSealError, match="non-integer"):
        seal.validate_fit_ledger_payload(bad, amendment)
    bad = copy.deepcopy(ledger)
    bad["skipped_counts"]["skipped_total_decision_units"] -= 1
    with pytest.raises(seal.TargetFreeSealError, match="slot accounting"):
        seal.validate_fit_ledger_payload(bad, amendment)
    bad = copy.deepcopy(ledger)
    bad["internal_method_call_counts"].pop("total_method_calls")
    with pytest.raises(seal.TargetFreeSealError, match="schema"):
        seal.validate_fit_ledger_payload(bad, amendment)
    bad = copy.deepcopy(ledger)
    bad["unit_slots"][0]["skip_reason"] = "UNFROZEN_SKIP_REASON"
    with pytest.raises(seal.TargetFreeSealError, match="frozen exact enum"):
        seal.validate_fit_ledger_payload(bad, amendment)
    bad = copy.deepcopy(ledger)
    bad["status"] = amendment["output_contract"]["decision_status_literals_exact"][
        "clear_no_family"
    ]
    with pytest.raises(seal.TargetFreeSealError, match="status/decision"):
        seal.validate_fit_ledger_payload(
            bad, amendment, expected_status=expected_status
        )


def test_runner_stdout_full_pretty_capture_and_ledger_derived_threadpool_chain(
    amendment: Mapping[str, Any],
) -> None:
    ledger = _fit_ledger_payload(amendment)
    row = ledger["unit_slots"][0]
    row.update(
        {
            "unit_status": "COMPLETED",
            "skip_reason": None,
            "pipeline_fit_calls": 1,
            "standard_scaler_fit_calls": 1,
            "ridge_fit_calls": 1,
            "predict_calls": 1,
            "fit_completed": True,
        }
    )
    ledger["executed_counts"].update(
        {
            "completed_primary_model_units": 1,
            "completed_total_decision_units": 1,
        }
    )
    ledger["skipped_counts"].update(
        {
            "skipped_primary_model_units": 71,
            "skipped_total_decision_units": 1_331,
        }
    )
    ledger["internal_method_call_counts"].update(
        {
            "sklearn.pipeline.Pipeline.fit": 1,
            "sklearn.preprocessing.StandardScaler.fit": 1,
            "sklearn.linear_model.Ridge.fit": 1,
            "total_method_calls": 3,
        }
    )
    decision = {
        "status": ledger["status"],
        "selected_family": amendment["family_decision_and_salvage"][
            "fixed_priority_if_both_pass"
        ][0],
        "global_ambiguity": False,
    }
    staged = _staged_identities()
    attempt = "target_free_duplicate_v3__20260811T000000000000Z"
    payload = _runner_stdout(
        amendment, staged=staged, decision=decision, attempt_id=attempt
    )
    response = amendment["planned_fit_budget"]["response_order"][0]
    payload["threadpool_info_before_inside_after"] = _threadpool_for_labels(
        amendment,
        [
            f"{response}/2022_H1/RIDGE_PIPELINE/FIT",
            f"{response}/2022_H1/RIDGE_PIPELINE/PREDICT",
        ],
    )
    assert seal.validate_runner_stdout(
        payload,
        attempt_id=attempt,
        amendment=amendment,
        staged_output_identities=staged,
        expected_decision=decision,
        fit_ledger=ledger,
    ) == payload
    assert payload["threadpool_info_before_inside_after"]["capture_counts"] == {
        "BEFORE_EXPLICIT_CONTEXT": 1,
        "INSIDE_CONTEXT": 2,
        "AFTER_CONTEXT": 2,
    }
    bad = copy.deepcopy(payload)
    bad["threadpool_info_before_inside_after"]["event_chain_sha256"] = "0" * 64
    with pytest.raises(seal.TargetFreeSealError, match="ledger-derived event chain"):
        seal.validate_runner_stdout(
            bad,
            attempt_id=attempt,
            amendment=amendment,
            staged_output_identities=staged,
            expected_decision=decision,
            fit_ledger=ledger,
        )
    bad = copy.deepcopy(payload)
    bad["staged_output_identities"].reverse()
    with pytest.raises(seal.TargetFreeSealError, match="staged7"):
        seal.validate_runner_stdout(
            bad,
            attempt_id=attempt,
            amendment=amendment,
            staged_output_identities=staged,
            expected_decision=decision,
            fit_ledger=ledger,
        )
    bad = copy.deepcopy(payload)
    bad["amendment_v4"] = seal.amendment_v4_identity()
    with pytest.raises(seal.TargetFreeSealError, match="runner stdout schema"):
        seal.validate_runner_stdout(
            bad,
            attempt_id=attempt,
            amendment=amendment,
            staged_output_identities=staged,
            expected_decision=decision,
            fit_ledger=ledger,
        )


def test_distribution_envelope_is_bound_to_decision_status_scope_and_diagnostics(
    amendment: Mapping[str, Any],
) -> None:
    schema = amendment["output_contract"]["output_record_schemas_exact"][
        "TARGET_FREE_DISTRIBUTION_STABILITY_V3_json"
    ]
    expected_status = amendment["output_contract"]["decision_status_literals_exact"][
        "positive"
    ]
    distribution = {
        "schema_version": 3,
        "artifact_type": "TARGET_FREE_DISTRIBUTION_STABILITY_V3",
        "status": expected_status,
        "scope_order": list(
            amendment["distribution_stability_contract"]["decision_scopes_exact"]
        ),
        "diagnostic_order": [
            *amendment["planned_fit_budget"]["response_order"],
            *seal.ENDPOINT_DIAGNOSTICS,
        ],
        "endpoint_pooled_metric_records": [
            {key: None for key in schema["endpoint_pooled_metric_record_keys_exact"]}
            for _ in range(3)
        ],
        "records": [
            {key: None for key in schema["record_keys_exact"]}
            for _ in range(624)
        ],
    }
    assert seal.validate_distribution_payload(
        distribution, amendment, expected_status=expected_status
    ) == distribution
    bad = copy.deepcopy(distribution)
    bad["status"] = amendment["output_contract"]["decision_status_literals_exact"][
        "clear_no_family"
    ]
    with pytest.raises(seal.TargetFreeSealError, match="status/decision"):
        seal.validate_distribution_payload(
            bad, amendment, expected_status=expected_status
        )
    bad = copy.deepcopy(distribution)
    bad["scope_order"][0] = "LEXICAL_OR_UNFROZEN_SCOPE"
    with pytest.raises(seal.TargetFreeSealError, match="scope order"):
        seal.validate_distribution_payload(
            bad, amendment, expected_status=expected_status
        )
    bad = copy.deepcopy(distribution)
    bad["records"][0]["invalid_reason"] = "UNFROZEN_INVALID_REASON"
    with pytest.raises(seal.TargetFreeSealError, match="frozen exact enum"):
        seal.validate_distribution_payload(
            bad, amendment, expected_status=expected_status
        )


@pytest.mark.parametrize("status_key", ["positive", "clear_no_family", "global_ambiguity"])
def test_auditor_stdout_exact17_no_fit_and_dynamic_decision_contract(
    amendment: Mapping[str, Any], status_key: str
) -> None:
    payload, authorization, independent_go, code = _auditor_stdout(
        amendment, status_key=status_key
    )
    assert seal.validate_auditor_stdout(
        payload,
        attempt_id=payload["attempt_id"],
        amendment=amendment,
        amendment_record=seal.amendment_identity(),
        amendment_v4_record=seal.amendment_v4_identity(),
        authorization_record=authorization,
        go_record=independent_go,
        code_identities=code,
        staged_output_identities=payload["staged_output_identities"],
    ) == payload


def test_auditor_stdout_rejects_count_schema_decision_and_no_fit_drift(
    amendment: Mapping[str, Any]
) -> None:
    payload, authorization, independent_go, code = _auditor_stdout(amendment)

    def validate(candidate: Mapping[str, Any]) -> None:
        seal.validate_auditor_stdout(
            candidate,
            attempt_id=payload["attempt_id"],
            amendment=amendment,
            amendment_record=seal.amendment_identity(),
            amendment_v4_record=seal.amendment_v4_identity(),
            authorization_record=authorization,
            go_record=independent_go,
            code_identities=code,
            staged_output_identities=payload["staged_output_identities"],
        )

    bad = copy.deepcopy(payload)
    bad["output_counts"]["derived_rows"] -= 1
    with pytest.raises(seal.TargetFreeSealError, match="output_counts"):
        validate(bad)
    bad = copy.deepcopy(payload)
    bad["decision"]["selected_family"] = "UNFROZEN_FAMILY"
    with pytest.raises(seal.TargetFreeSealError, match="decision"):
        validate(bad)
    bad = copy.deepcopy(payload)
    bad["models_fit"] = 1
    with pytest.raises(seal.TargetFreeSealError, match="fit"):
        validate(bad)
    bad = copy.deepcopy(payload)
    bad["poststate"]["manifest_present"] = True
    with pytest.raises(seal.TargetFreeSealError, match="poststate"):
        validate(bad)
    bad = copy.deepcopy(payload)
    bad["output_counts"]["aliases"] = 2.0
    with pytest.raises(seal.TargetFreeSealError, match="output_counts"):
        validate(bad)
    bad = copy.deepcopy(payload)
    bad["models_fit"] = False
    with pytest.raises(seal.TargetFreeSealError, match="fit"):
        validate(bad)
    bad = copy.deepcopy(payload)
    bad["amendment_v4"]["canonical_sha256"] = "0" * 64
    with pytest.raises(seal.TargetFreeSealError, match="V4 amendment"):
        validate(bad)


def test_audit_result_binds_full_stdout_bytes_command_empty_stderr_and_zero_fit(
    tmp_path: Path, amendment: Mapping[str, Any]
) -> None:
    payload, authorization, independent_go, code = _auditor_stdout(amendment)
    postrun = {
        "attempt_id": payload["attempt_id"],
        "amendment": seal.amendment_identity(),
        "amendment_v4": seal.amendment_v4_identity(),
        "authorization": authorization,
        "independent_go": independent_go,
        "code_identities": code,
        "staged_output_identities": payload["staged_output_identities"],
    }
    commands = seal.expected_required_commands(tmp_path, amendment)
    result = _process_capture(payload, commands["auditor"])
    runner_payload = _runner_stdout(
        amendment,
        staged=payload["staged_output_identities"],
        decision=payload["decision"],
        attempt_id=payload["attempt_id"],
    )
    result["runner_result"] = _process_capture(
        runner_payload, commands["runner"]
    )
    assert seal.validate_audit_result(
        result, postrun, amendment=amendment, artifact_root=tmp_path
    ) == result
    round_tripped = seal.strict_json_loads(
        seal.pretty_json_bytes(result).decode("utf-8"),
        label="round-tripped postrun audit_result",
    )
    seal.validate_audit_result(
        round_tripped, postrun, amendment=amendment, artifact_root=tmp_path
    )
    bad = copy.deepcopy(result)
    bad["stdout_size_bytes"] += 1
    with pytest.raises(seal.TargetFreeSealError, match="stdout size"):
        seal.validate_audit_result(
            bad, postrun, amendment=amendment, artifact_root=tmp_path
        )
    encoded = seal.pretty_json_bytes(payload)
    compact = seal.compact_json_bytes(payload) + b"\n"
    assert compact != encoded
    bad = copy.deepcopy(result)
    bad["stdout_size_bytes"] = len(compact)
    bad["stdout_sha256"] = _sha(compact)
    with pytest.raises(seal.TargetFreeSealError, match="stdout size"):
        seal.validate_audit_result(
            bad, postrun, amendment=amendment, artifact_root=tmp_path
        )
    bad = copy.deepcopy(result)
    bad["exit_code"] = False
    with pytest.raises(seal.TargetFreeSealError, match="exit code"):
        seal.validate_audit_result(
            bad, postrun, amendment=amendment, artifact_root=tmp_path
        )
    bad = copy.deepcopy(result)
    bad["runner_result"]["stdout_payload"]["attempt_id"] = (
        "target_free_duplicate_v3__20260811T000000000001Z"
    )
    with pytest.raises(seal.TargetFreeSealError, match="runner stdout attempt"):
        seal.validate_audit_result(
            bad, postrun, amendment=amendment, artifact_root=tmp_path
        )
    bad = copy.deepcopy(result)
    bad["runner_result"]["stdout_size_bytes"] += 1
    with pytest.raises(seal.TargetFreeSealError, match="runner stdout size"):
        seal.validate_audit_result(
            bad, postrun, amendment=amendment, artifact_root=tmp_path
        )


def test_code_seal_exact12_dual_bound_payload_and_tmp_only_publication(
    tmp_path: Path, amendment: Mapping[str, Any]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    code = _make_code_repo(repo)
    evidence = _test_evidence(repo, amendment)
    payload = seal.build_code_seal_payload(
        amendment=amendment,
        code_identities=code,
        test_evidence=evidence,
        created_utc="2026-08-11T16:00:01.000000Z",
        repo_root=repo,
    )
    assert set(payload) == seal.CONTROL_SPECS["code_seal"]["keys"]
    assert len(payload) == 12
    assert payload["amendment"] == seal.amendment_identity()
    assert payload["amendment_v4"] == seal.amendment_v4_identity()
    assert payload["authority_scope"] == seal.DOCUMENTARY_SCOPE
    assert payload["required_next_controls"] == seal.REQUIRED_NEXT_CONTROLS

    artifact = tmp_path / "artifact"
    (artifact / "prereg").mkdir(parents=True)
    identity = seal.publish_code_seal(
        artifact,
        payload,
        amendment=amendment,
        repo_root=repo,
    )
    destination = artifact / seal.CODE_SEAL_RELATIVE
    assert identity == seal.file_identity(destination, relative_to=artifact)
    with pytest.raises(seal.TargetFreeSealError, match="already exists"):
        seal.publish_code_seal(
            artifact,
            payload,
            amendment=amendment,
            repo_root=repo,
        )
    assert {path.name for path in destination.parent.iterdir()} == {destination.name}

    bad = copy.deepcopy(payload)
    bad.pop("amendment_v4")
    with pytest.raises(seal.TargetFreeSealError, match="top-level schema"):
        seal.validate_control_payload(
            "code_seal", bad, amendment=amendment, repo_root=repo
        )
    bad = copy.deepcopy(payload)
    bad["created_utc"] = "2026-08-11T15:00:00.000000Z"
    with pytest.raises(seal.TargetFreeSealError, match="V4 documentary correction"):
        seal.validate_control_payload(
            "code_seal", bad, amendment=amendment, repo_root=repo
        )


def test_test_evidence_is_exact_argv_and_rejects_bool_counts(
    tmp_path: Path, amendment: Mapping[str, Any]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_code_repo(repo)
    evidence = _test_evidence(repo, amendment)
    assert seal.validate_test_evidence(
        evidence, repo_root=repo, amendment=amendment
    ) == evidence
    bad = copy.deepcopy(evidence)
    bad["commands"]["sealer"].append("tests/another_test.py")
    with pytest.raises(seal.TargetFreeSealError, match="commands"):
        seal.validate_test_evidence(bad, repo_root=repo, amendment=amendment)
    bad = copy.deepcopy(evidence)
    bad["results"]["runner"]["passed"] = True
    with pytest.raises(seal.TargetFreeSealError, match="count"):
        seal.validate_test_evidence(bad, repo_root=repo, amendment=amendment)
    bad = copy.deepcopy(evidence)
    bad["results"]["runner"]["exit_code"] = False
    with pytest.raises(seal.TargetFreeSealError, match="exit code"):
        seal.validate_test_evidence(bad, repo_root=repo, amendment=amendment)
    bad = copy.deepcopy(evidence)
    bad["results"]["runner"]["summary"] = "2 passed in 0.01s"
    with pytest.raises(seal.TargetFreeSealError, match="disagree"):
        seal.validate_test_evidence(bad, repo_root=repo, amendment=amendment)
    rounded = copy.deepcopy(evidence)
    rounded["results"]["runner"]["summary"] = "1 passed in 61.00s (0:01:00)"
    seal.validate_test_evidence(rounded, repo_root=repo, amendment=amendment)
    no_suffix = copy.deepcopy(evidence)
    no_suffix["results"]["runner"]["summary"] = "1 passed in 60.00s"
    seal.validate_test_evidence(no_suffix, repo_root=repo, amendment=amendment)
    for invalid in (
        "1 passed in 61.01s (0:01:00)",
        "1 passed in 61.00s (0:00:59)",
        "1 passed in 61.00s (00:01:00)",
        "1 passed in 61.00s (0:01:02)",
    ):
        bad = copy.deepcopy(evidence)
        bad["results"]["runner"]["summary"] = invalid
        with pytest.raises(seal.TargetFreeSealError, match="summary"):
            seal.validate_test_evidence(
                bad, repo_root=repo, amendment=amendment
            )


def test_authorization_and_go_contracts_use_frozen_exact14_budget_and_windows_argv(
    tmp_path: Path, amendment: Mapping[str, Any]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    code = _make_code_repo(repo)
    root = tmp_path / "artifact"
    root.mkdir()
    attempt = "target_free_duplicate_v3__20260811T000000000000Z"
    common = {
        "schema_version": 3,
        "created_utc": "2026-08-11T16:00:02.000000Z",
        "attempt_id": attempt,
        "amendment": seal.amendment_identity(),
        "amendment_v4": seal.amendment_v4_identity(),
        "code_seal": _identity(seal.CODE_SEAL_RELATIVE),
        "code_identities": code,
        "output_namespace": dict(seal.OUTPUT_NAMESPACE),
        "required_commands": seal.expected_required_commands(root, amendment),
    }
    authorization = {
        **common,
        "artifact_type": seal.CONTROL_SPECS["authorization"]["artifact_type"],
        "status": seal.CONTROL_SPECS["authorization"]["status"],
        "execution_budget": dict(seal.EXECUTION_BUDGET),
        "required_review": dict(seal.REQUIRED_REVIEW),
        "required_go": dict(seal.REQUIRED_GO),
        "authority_scope": dict(seal.AUTHORIZATION_SCOPE),
        "publisher": seal.CONTROL_SPECS["authorization"]["publisher"],
    }
    assert len(authorization["execution_budget"]) == 14
    seal.validate_control_payload(
        "authorization",
        authorization,
        amendment=amendment,
        artifact_root=root,
        repo_root=repo,
    )
    go = {
        **common,
        "artifact_type": seal.CONTROL_SPECS["go"]["artifact_type"],
        "status": seal.CONTROL_SPECS["go"]["status"],
        "authorization": _identity(seal.AUTHORIZATION_RELATIVE, marker="b"),
        "independent_review": _identity(seal.REVIEW_RELATIVE, marker="c"),
        "authority_scope": dict(seal.GO_SCOPE),
        "single_attempt": dict(seal.SINGLE_ATTEMPT),
        "publisher": seal.CONTROL_SPECS["go"]["publisher"],
    }
    seal.validate_control_payload(
        "go", go, amendment=amendment, artifact_root=root, repo_root=repo
    )
    assert "\\" in authorization["required_commands"]["runner"][0]
    bad = copy.deepcopy(authorization)
    bad["execution_budget"]["alias_count"] = 3
    with pytest.raises(seal.TargetFreeSealError, match="budget"):
        seal.validate_control_payload(
            "authorization",
            bad,
            amendment=amendment,
            artifact_root=root,
            repo_root=repo,
        )
    bad = copy.deepcopy(authorization)
    bad.pop("amendment_v4")
    with pytest.raises(seal.TargetFreeSealError, match="top-level schema"):
        seal.validate_control_payload(
            "authorization",
            bad,
            amendment=amendment,
            artifact_root=root,
            repo_root=repo,
        )
    bad = copy.deepcopy(authorization)
    bad["amendment_v4"]["sha256"] = "0" * 64
    with pytest.raises(seal.TargetFreeSealError, match="V4 amendment"):
        seal.validate_control_payload(
            "authorization",
            bad,
            amendment=amendment,
            artifact_root=root,
            repo_root=repo,
        )
    bad = copy.deepcopy(authorization)
    bad["execution_budget"]["attempt_count"] = True
    with pytest.raises(seal.TargetFreeSealError, match="budget"):
        seal.validate_control_payload(
            "authorization",
            bad,
            amendment=amendment,
            artifact_root=root,
            repo_root=repo,
        )
    bad = copy.deepcopy(authorization)
    bad["execution_budget"]["attempt_number"] = bad["execution_budget"].pop(
        "attempt_count"
    )
    with pytest.raises(seal.TargetFreeSealError, match="budget"):
        seal.validate_control_payload(
            "authorization",
            bad,
            amendment=amendment,
            artifact_root=root,
            repo_root=repo,
        )


def test_csv_validator_is_bounded_exact_shape_and_rejects_nonfinite_text(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.csv"
    good.write_text("a,b\n1,2\n3,4\n", encoding="utf-8", newline="")
    seal.validate_csv_shape(good, columns=("a", "b"), rows=2, label="good")
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,+Infinity\n", encoding="utf-8", newline="")
    with pytest.raises(seal.TargetFreeSealError, match="nonfinite"):
        seal.validate_csv_shape(bad, columns=("a", "b"), rows=1, label="bad")
    bad.write_bytes(b"a,b\r\n1,2\r\n")
    with pytest.raises(seal.TargetFreeSealError, match="CR/NUL"):
        seal.validate_csv_shape(bad, columns=("a", "b"), rows=1, label="bad")
    bad.write_text("a,b\n1,1e400\n", encoding="utf-8", newline="")
    with pytest.raises(seal.TargetFreeSealError, match="overflows"):
        seal.validate_csv_shape(bad, columns=("a", "b"), rows=1, label="bad")
    reason = tmp_path / "reason.csv"
    reason.write_text(
        "valid,invalid_reason\nFALSE,UNFROZEN_INVALID_REASON\n",
        encoding="utf-8",
        newline="",
    )
    with pytest.raises(seal.TargetFreeSealError, match="frozen exact enum"):
        seal.validate_csv_shape(
            reason,
            columns=("valid", "invalid_reason"),
            rows=1,
            label="reason",
            invalid_reason_enum=("STRUCTURALLY_NOT_APPLICABLE",),
        )


def test_v4_csv_boolean_and_boundary_flag_grammar_exact_positive_and_negatives(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v4-grammar.csv"
    columns = ("valid", "nullable_gate", "boundary_equality_flags")
    cutoff_order = tuple(seal.CUTOFF_VALUES)

    def validate_row(valid: str, nullable: str, flags: str) -> None:
        path.write_text(
            ",".join(columns) + "\n" + ",".join((valid, nullable, flags)) + "\n",
            encoding="utf-8",
            newline="",
        )
        seal.validate_csv_shape(
            path,
            columns=columns,
            rows=1,
            label="V4 CSV grammar",
            boolean_columns={"valid": False, "nullable_gate": True},
            boundary_flags_column="boundary_equality_flags",
            boundary_flag_order=cutoff_order,
        )

    validate_row("TRUE", "", "")
    validate_row("FALSE", "FALSE", f"{cutoff_order[0]};{cutoff_order[3]}")
    for valid, nullable, flags, message in (
        ("false", "TRUE", "", "uppercase TRUE/FALSE"),
        ("1", "TRUE", "", "uppercase TRUE/FALSE"),
        ("TRUE", "FALSE", "FALSE", "boundary flags"),
        ("TRUE", "FALSE", f"{cutoff_order[0]}=TRUE", "boundary flags"),
        (
            "TRUE",
            "FALSE",
            f"{cutoff_order[3]};{cutoff_order[0]}",
            "boundary flags",
        ),
        (
            "TRUE",
            "FALSE",
            f"{cutoff_order[0]};{cutoff_order[0]}",
            "boundary flags",
        ),
        ("TRUE", "FALSE", "UNFROZEN_CUTOFF", "boundary flags"),
    ):
        with pytest.raises(seal.TargetFreeSealError, match=message):
            validate_row(valid, nullable, flags)


def test_static_self_validator_proves_single_publisher_no_fit_no_network_no_parquet_values() -> None:
    report = seal.static_self_validate()
    assert report["status"] == "PASS_V3_SEALER_STATIC_SELF_VALIDATION"
    assert report["os_link_call_count"] == 1
    assert report["overwrite_capable_calls"] == 0
    assert report["network_calls"] == 0
    assert report["parquet_logical_value_reads"] == 0
    assert report["parquet_footer_metadata_reads"] == 1
    assert report["models_fit"] == 0
    assert report["predictions_computed"] == 0

    source_path = Path(seal.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    owners: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(
            isinstance(child, ast.Call) and ast.unparse(child.func) == "os.link"
            for child in ast.walk(node)
        ):
            owners.append(node.name)
    assert owners == ["publish_staged_create_if_absent"]


def test_network_guard_callback_is_fail_closed_without_making_a_request() -> None:
    with pytest.raises(seal.TargetFreeSealError, match="network"):
        seal._network_forbidden("fixture.invalid")


def test_actual_execution_argv_is_type_exact_and_byte_for_byte_authorized(
    amendment: Mapping[str, Any],
) -> None:
    expected = seal.expected_required_commands(
        seal.ARTIFACT_ROOT_DEFAULT, amendment
    )["sealer"]
    assert seal.validate_actual_execution_argv(list(expected), expected) == expected

    extra = [*expected, "--unexpected"]
    reordered = list(expected)
    reordered[4], reordered[6] = reordered[6], reordered[4]
    relative_interpreter = list(expected)
    relative_interpreter[0] = Path(expected[0]).name
    relative_root = list(expected)
    relative_root[5] = "artifacts/relative-root"
    relative_authorization = list(expected)
    relative_authorization[7] = seal.AUTHORIZATION_RELATIVE
    relative_postrun = list(expected)
    relative_postrun[-1] = seal.POSTRUN_RELATIVE
    missing_postrun = list(expected[:-2])
    nonstring = list(expected)
    nonstring[0] = 1  # type: ignore[assignment]

    for observed in (
        extra,
        reordered,
        relative_interpreter,
        relative_root,
        relative_authorization,
        relative_postrun,
        missing_postrun,
        nonstring,
        tuple(expected),
    ):
        with pytest.raises(
            seal.TargetFreeSealError, match="actual sealer sys.orig_argv"
        ):
            seal.validate_actual_execution_argv(observed, expected)


def test_actual_execution_argv_guard_precedes_every_staged_output_read() -> None:
    source_path = Path(seal.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_authority_chain"
    )
    calls = {
        ast.unparse(node.func): node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    }
    guard_line = calls["validate_actual_execution_argv"]
    assert calls["validate_control_payload"] < guard_line
    assert guard_line < calls["validate_postrun_namespace_state"]
    assert guard_line < calls["validate_staged_csv_outputs"]
    assert guard_line < calls["validate_staged_json_outputs"]
    function_source = ast.get_source_segment(
        source_path.read_text(encoding="utf-8"), function
    )
    assert function_source is not None
    assert 'artifact_root=None if kind == "postrun" else root' in function_source


def test_main_final_mode_dispatches_validated_chain_then_finalizer_without_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "artifact"
    repo = tmp_path / "repo"
    root.mkdir()
    repo.mkdir()
    authorization = root / seal.AUTHORIZATION_RELATIVE
    independent_go = root / seal.GO_RELATIVE
    postrun = root / seal.POSTRUN_RELATIVE
    calls: list[tuple[Any, ...]] = []
    validation = {"fixture": "validated"}
    actual_orig_argv = ["C:\\exact\\python.exe", "-B", "-m", "fixture.sealer"]

    monkeypatch.setattr(seal, "install_network_guard", lambda: None)
    monkeypatch.setattr(seal.sys, "orig_argv", actual_orig_argv)
    monkeypatch.setattr(
        seal,
        "_parse_args",
        lambda _argv: seal.argparse.Namespace(
            root=root,
            authorization=authorization,
            independent_go=independent_go,
            postrun_pass=postrun,
            code_seal_test_evidence=None,
            repo_root=repo,
        ),
    )

    def validate_fixture(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        calls.append((*args, kwargs))
        return validation

    def finalize_fixture(
        observed_root: Path, observed_validation: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        assert observed_root == root.resolve()
        assert observed_validation is validation
        calls.append(("finalize",))
        return {"status": "PASS_FIXTURE", "models_fit": 0}

    monkeypatch.setattr(seal, "validate_authority_chain", validate_fixture)
    monkeypatch.setattr(
        seal, "finalize_validated_authority_chain", finalize_fixture
    )
    assert seal.main() == 0
    assert calls[0][:4] == (
        root.resolve(),
        authorization.resolve(),
        independent_go.resolve(),
        postrun.resolve(),
    )
    assert calls[0][4] == {
        "repo_root": repo.resolve(),
        "actual_orig_argv": actual_orig_argv,
    }
    assert calls[1] == ("finalize",)
    assert capsys.readouterr().out == '{"models_fit": 0, "status": "PASS_FIXTURE"}\n'


def test_main_code_seal_mode_remains_a_separate_valid_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "artifact"
    repo = tmp_path / "repo"
    evidence = tmp_path / "test-evidence.json"
    actual_orig_argv = [
        "C:\\exact\\python.exe",
        "-B",
        "-m",
        "scripts.seal_noaa_gfs_target_free_duplicate_v3",
        "--root",
        str(root),
        "--code-seal-test-evidence",
        str(evidence),
        "--repo-root",
        str(repo),
    ]
    calls: list[tuple[Path, Path, Path]] = []

    monkeypatch.setattr(seal.sys, "orig_argv", actual_orig_argv)

    def code_seal_fixture(
        observed_root: Path, observed_repo: Path, observed_evidence: Path
    ) -> Mapping[str, Any]:
        calls.append((observed_root, observed_repo, observed_evidence))
        return {"status": "PASS_CODE_SEAL_FIXTURE"}

    monkeypatch.setattr(seal, "code_seal_mode", code_seal_fixture)
    monkeypatch.setattr(
        seal,
        "validate_authority_chain",
        lambda *_args, **_kwargs: pytest.fail(
            "execution-mode validation entered from code-seal mode"
        ),
    )
    assert seal.main(
        [
            "--root",
            str(root),
            "--code-seal-test-evidence",
            str(evidence),
            "--repo-root",
            str(repo),
        ]
    ) == 0
    assert calls == [
        (root.resolve(), repo.resolve(), evidence.resolve())
    ]
    assert capsys.readouterr().out == '{"status": "PASS_CODE_SEAL_FIXTURE"}\n'


def test_main_injected_final_argv_has_no_production_execution_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    authorization = root / seal.AUTHORIZATION_RELATIVE
    independent_go = root / seal.GO_RELATIVE
    postrun = root / seal.POSTRUN_RELATIVE
    monkeypatch.setattr(seal, "install_network_guard", lambda: None)

    def reject_test_seam(*_args: Any, **kwargs: Any) -> Mapping[str, Any]:
        assert kwargs["actual_orig_argv"] is None
        seal.validate_actual_execution_argv(
            kwargs["actual_orig_argv"], ["C:\\exact\\python.exe"]
        )
        raise AssertionError("unreachable")

    monkeypatch.setattr(seal, "validate_authority_chain", reject_test_seam)
    monkeypatch.setattr(
        seal,
        "finalize_validated_authority_chain",
        lambda *_args, **_kwargs: pytest.fail("test seam reached final publication"),
    )
    with pytest.raises(
        seal.TargetFreeSealError, match="actual sealer sys.orig_argv"
    ):
        seal.main(
            [
                "--root",
                str(root),
                "--authorization",
                str(authorization),
                "--independent-go",
                str(independent_go),
                "--postrun-pass",
                str(postrun),
            ]
        )


def test_only_authorized_pair_is_touched_by_this_change() -> None:
    assert Path(seal.__file__).name == "seal_noaa_gfs_target_free_duplicate_v3.py"
    assert Path(__file__).name == "test_seal_noaa_gfs_target_free_duplicate_v3.py"
    assert set(seal.CODE_ROLE_PATHS) == {
        "runner",
        "runner_test",
        "auditor",
        "auditor_test",
        "sealer",
        "sealer_test",
    }

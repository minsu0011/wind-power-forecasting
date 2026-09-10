from __future__ import annotations

import ast
import builtins
import copy
import datetime as dt
import gc
import inspect
import json
import weakref
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from scripts import run_noaa_gfs_target_free_duplicate_v5 as subject


def valid_control(**overrides: bool) -> subject.ControlValidation:
    values = {
        field.name: True
        for field in subject.dataclasses.fields(subject.ControlValidation)
    }
    values.update(overrides)
    return subject.ControlValidation(**values)


def metric(r2: float, nrmse: float, *, rows: int = 48) -> subject.MetricResult:
    return subject.MetricResult(
        rows=rows,
        valid=True,
        invalid_reason=None,
        unique_count=3,
        sst=np.float64(1.0),
        sse=np.float64(1.0),
        rmse=np.float64(nrmse),
        r2=np.float64(r2),
        nrmse=np.float64(nrmse),
        denominator=np.float64(1.0),
    )


def estimator(
    name: str, rmse: float, r2: float, nrmse: float
) -> subject.EstimatorMetric:
    return subject.EstimatorMetric(
        name, np.float64(rmse), np.float64(r2), np.float64(nrmse)
    )


def blank_affine_screens() -> list[subject.AffineScreen]:
    return [
        subject.AffineScreen(name, None, None, None, False)
        for name in subject.PREDICTORS
    ]


def component(classification: str) -> subject.ComponentDecision:
    return subject.ComponentDecision(
        classification=classification,
        counts_as_nonredundant=classification == "PASS_NONREDUNDANT_MARGIN",
        independent_estimator="RIDGE_PIPELINE",
        parent_reference_estimators=("EXTRA_TREES",),
        affine_hits=(),
    )


def raw_result(
    response: str,
    classification: str = "CLEAR_NONNOVEL_CONSTANT",
) -> dict[str, object]:
    premodel = classification in subject.PREMODEL_CLASSIFICATIONS
    ambiguous = classification in subject.AMBIGUOUS_CLASSIFICATIONS
    no_selector = premodel or (
        classification == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
    )
    return {
        "response": response,
        "classification": classification,
        "counts_as_nonredundant": classification == "PASS_NONREDUNDANT_MARGIN",
        "independent_estimator": None if no_selector else "RIDGE_PIPELINE",
        "parent_reference_estimators": [] if no_selector else ["EXTRA_TREES"],
        "parent_redundancy_boolean": None if premodel or ambiguous else False,
        "affine_qualifying_predictors": [],
        "cell_stability_pass": None if premodel or ambiguous else True,
        "distribution_stability_pass": None if premodel or ambiguous else True,
        "boundary_equality_flags": (
            ["INDEPENDENT_R2_STABLE_MARGIN_0.9925"]
            if classification == "AMBIGUOUS_THRESHOLD_EQUALITY"
            else []
        ),
        "ambiguity_reason": classification if ambiguous else None,
    }


def endpoint_result(
    diagnostic: str,
    raw: dict[str, dict[str, object]],
    verdict: str = "CLEAR_ENDPOINT_VETO_FAILURE",
) -> dict[str, object]:
    binding = {
        response: raw[response]["independent_estimator"]
        for response in subject._endpoint_source_responses(diagnostic)
    }
    source_null = any(value is None for value in binding.values())
    if source_null:
        aggregates: tuple[object, object, object] = (None, None, None)
    else:
        passed = verdict == "PASS_ENDPOINT_NONREDUNDANT_MARGIN"
        aggregates = (passed, passed, passed)
    return {
        "diagnostic": diagnostic,
        "verdict": verdict,
        "source_component_estimator_binding": binding,
        "pooled_stable_margin_pass": aggregates[0],
        "cell_stability_pass": aggregates[1],
        "distribution_stability_pass": aggregates[2],
        "boundary_equality_flags": [],
        "clear_pass": verdict == "PASS_ENDPOINT_NONREDUNDANT_MARGIN",
        "ambiguity_reason": (
            verdict if verdict.startswith("AMBIGUOUS_") else None
        ),
    }


def decision_document(
    raw_rows: list[dict[str, object]] | None = None,
    *,
    endpoints_pass: bool = False,
) -> dict[str, object]:
    rows = raw_rows or [raw_result(response) for response in subject.RESPONSES]
    by_response = {str(row["response"]): row for row in rows}
    endpoint_verdict = (
        "PASS_ENDPOINT_NONREDUNDANT_MARGIN"
        if endpoints_pass
        else "CLEAR_ENDPOINT_VETO_FAILURE"
    )
    endpoints = [
        endpoint_result(diagnostic, by_response, endpoint_verdict)
        for diagnostic in subject.ENDPOINT_VETO_DIAGNOSTICS
    ]
    pbl = rows[0]["classification"] == "PASS_NONREDUNDANT_MARGIN"
    passed_wind = [
        str(row["response"])
        for row in rows[1:]
        if row["classification"] == "PASS_NONREDUNDANT_MARGIN"
    ]
    wind = (
        len(passed_wind) >= 4
        and {"UGRD", "VGRD"}.issubset(
            {name.split("_", 1)[0] for name in passed_wind}
        )
        and len({name.split("_", 1)[1] for name in passed_wind}) >= 2
        and all(bool(row["clear_pass"]) for row in endpoints)
    )
    selected = "LOW_LEVEL_ISOBARIC_WIND_PROFILE" if wind else (
        "PBL_HEIGHT" if pbl else None
    )
    status = (
        "PASS_ONE_TARGET_FREE_FAMILY_SELECTED_PENDING_INDEPENDENT_POSTRUN_SEAL"
        if selected
        else "STOP_NO_TARGET_FREE_INCREMENTAL_FAMILY"
    )
    return {
        "schema_version": 5,
        "artifact_type": "TARGET_FREE_FAMILY_DECISION_V5",
        "status": status,
        "raw_component_order": list(subject.RESPONSES),
        "raw_component_results": rows,
        "endpoint_veto_results": endpoints,
        "global_ambiguity": False,
        "global_ambiguity_reasons": [],
        "family_clear_pass_results": {
            "LOW_LEVEL_ISOBARIC_WIND_PROFILE": wind,
            "PBL_HEIGHT": pbl,
        },
        "fixed_priority": list(subject.FIXED_FAMILY_PRIORITY),
        "selected_family": selected,
        "selected_raw_columns": (
            list(subject.WIND_RESPONSES)
            if selected == "LOW_LEVEL_ISOBARIC_WIND_PROFILE"
            else list(subject.PBL_RESPONSES) if selected == "PBL_HEIGHT" else []
        ),
        "selected_derived_columns": (
            list(subject.DERIVED_DIAGNOSTICS)
            if selected == "LOW_LEVEL_ISOBARIC_WIND_PROFILE"
            else []
        ),
        "all_threshold_float_hex_witnesses": [],
        "all_tied_witnesses": [],
        "no_label_network_future_year_model_submission_attestation": dict(
            subject.FORBIDDEN_ACCESS_ATTESTATION
        ),
    }


def constant_raw_evaluations() -> dict[str, subject.ResponseEvaluation]:
    reason = "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3"
    pooled = subject.PooledDenominator(
        np.float64(0.0), np.float64(0.0), np.float64(1e-9)
    )
    result: dict[str, subject.ResponseEvaluation] = {}
    for response in subject.RESPONSES:
        invalid = subject._empty_metric_result(
            subject.EXPECTED_SITE_ROWS, pooled.denominator, reason
        )
        duplicate_records = tuple(
            [
                subject._duplicate_metric_record(
                    record_kind="POOLED_ESTIMATOR",
                    response=response,
                    estimator=estimator_name,
                    predictor=None,
                    metric=invalid,
                    pooled=pooled,
                    component_classification="CLEAR_NONNOVEL_CONSTANT",
                )
                for estimator_name in ("RIDGE_PIPELINE", "EXTRA_TREES")
            ]
            + [
                subject._duplicate_metric_record(
                    record_kind="AFFINE_PREDICTOR",
                    response=response,
                    estimator=None,
                    predictor=predictor,
                    metric=invalid,
                    pooled=pooled,
                    component_classification="CLEAR_NONNOVEL_CONSTANT",
                )
                for predictor in subject.PREDICTORS
            ]
        )
        result[response] = subject.ResponseEvaluation(
            response=response,
            actual=np.zeros(subject.EXPECTED_SITE_ROWS, dtype=np.float64),
            pooled=pooled,
            ridge_oof=None,
            extra_trees_oof=None,
            independent_oof=None,
            independent_estimator=None,
            raw_result=raw_result(response),
            duplicate_metric_records=duplicate_records,
            stability_records=subject.empty_stability_records(
                response,
                diagnostic_kind="RAW_RESPONSE",
                denominator=pooled.denominator,
                reason=reason,
            ),
            distribution_records=subject.empty_distribution_records(
                response, reason=reason
            ),
            threshold_witnesses=(),
            tied_witnesses=(),
            completed_fit_records=(),
            skip_reason=reason,
        )
    return result


def empty_derived_evaluation(
    raw_evaluations: dict[str, subject.ResponseEvaluation],
) -> subject.DerivedEvaluation:
    raw_rows = {
        response: dict(evaluation.raw_result)
        for response, evaluation in raw_evaluations.items()
    }
    endpoints = tuple(
        endpoint_result(diagnostic, raw_rows)
        for diagnostic in subject.ENDPOINT_VETO_DIAGNOSTICS
    )
    pooled_records = []
    for endpoint in endpoints:
        binding = endpoint["source_component_estimator_binding"]
        pooled_records.append(
            {
                "diagnostic": endpoint["diagnostic"],
                "rows": subject.EXPECTED_SITE_ROWS,
                "valid": False,
                "invalid_reason": "STRUCTURALLY_NOT_APPLICABLE",
                "source_component_estimator_binding": binding,
                "sse": None,
                "sse_float_hex": None,
                "rmse": None,
                "rmse_float_hex": None,
                "r2": None,
                "r2_float_hex": None,
                "nrmse": None,
                "nrmse_float_hex": None,
                "actual_q05": 0.0,
                "actual_q05_float_hex": "0x0.0p+0",
                "actual_q95": 0.0,
                "actual_q95_float_hex": "0x0.0p+0",
                "nrmse_denominator": 1e-9,
                "nrmse_denominator_float_hex": float(1e-9).hex().lower(),
                "r2_boundary_equal_0_9925": False,
                "nrmse_boundary_equal_0_025": False,
                "strict_stable_margin_pass": False,
                "endpoint_pooled_verdict": "CLEAR_ENDPOINT_VETO_FAILURE",
            }
        )
    return subject.DerivedEvaluation(
        site_actual={},
        site_prediction={},
        group_metadata=None,
        group_actual={},
        group_prediction={},
        endpoint_results=endpoints,
        endpoint_pooled_records=tuple(pooled_records),
        stability_records=tuple(
            item
            for diagnostic in subject.ENDPOINT_VETO_DIAGNOSTICS
            for item in subject.empty_stability_records(
                diagnostic,
                diagnostic_kind="ENDPOINT_VETO",
                denominator=np.float64(1e-9),
                reason="STRUCTURALLY_NOT_APPLICABLE",
            )
        ),
        distribution_records=tuple(
            item
            for diagnostic in subject.ENDPOINT_VETO_DIAGNOSTICS
            for item in subject.empty_distribution_records(
                diagnostic, reason="STRUCTURALLY_NOT_APPLICABLE"
            )
        ),
        threshold_witnesses=(),
    )


def synthetic_completed_fit_records(response: str) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for unit in subject.planned_fit_units():
        if unit.response != response:
            continue
        if unit.unit_kind == "PRIMARY_MODEL":
            ridge = unit.estimator_unit == "RIDGE_PIPELINE"
            records.append(
                {
                    "planned_unit_slot_id": unit.planned_unit_slot_id,
                    "decision_fit_ordinal": unit.decision_fit_ordinal,
                    "unit_kind": unit.unit_kind,
                    "response": response,
                    "fold": unit.fold,
                    "estimator_unit": unit.estimator_unit,
                    "training_rows": 14_688,
                    "training_columns": 35,
                    "heldout_rows": 4_896,
                    "input_dtype": (
                        "float64_C_CONTIGUOUS"
                        if ridge
                        else "float32_C_CONTIGUOUS"
                    ),
                    "response_dtype": "float64",
                    "unit_status": subject.UNIT_COMPLETED,
                    "skip_reason": None,
                    "pipeline_fit_calls": int(ridge),
                    "standard_scaler_fit_calls": int(ridge),
                    "ridge_fit_calls": int(ridge),
                    "extra_trees_fit_calls": int(not ridge),
                    "predict_calls": 1,
                    "random_state": None if ridge else 260810,
                    "fit_completed": True,
                }
            )
        else:
            sxx = 1.0
            slope = 0.1
            intercept = 0.0
            records.append(
                {
                    "planned_unit_slot_id": unit.planned_unit_slot_id,
                    "decision_fit_ordinal": unit.decision_fit_ordinal,
                    "analytic_affine_ordinal": (
                        unit.decision_fit_ordinal - subject.EXPECTED_PRIMARY_UNITS
                    ),
                    "unit_kind": unit.unit_kind,
                    "response": response,
                    "predictor": unit.predictor,
                    "fold": unit.fold,
                    "training_rows": 14_688,
                    "heldout_rows": 4_896,
                    "unit_status": subject.UNIT_COMPLETED,
                    "skip_reason": None,
                    "sxx": sxx,
                    "sxx_float_hex": float(sxx).hex().lower(),
                    "constant_predictor_branch": False,
                    "slope": slope,
                    "slope_float_hex": float(slope).hex().lower(),
                    "intercept": intercept,
                    "intercept_float_hex": float(intercept).hex().lower(),
                    "fit_completed": True,
                }
            )
    assert len(records) == 148
    return tuple(records)


def synthetic_valid_stability_records(
    diagnostic: str, *, diagnostic_kind: str, denominator: np.float64
) -> tuple[dict[str, object], ...]:
    records = subject.empty_stability_records(
        diagnostic,
        diagnostic_kind=diagnostic_kind,
        denominator=denominator,
        reason="STRUCTURALLY_NOT_APPLICABLE",
    )
    result: list[dict[str, object]] = []
    values = {
        "sst": 1.0,
        "sse": 0.1,
        "rmse": 0.1,
        "r2": 0.9,
        "nrmse": 0.1,
    }
    for source in records:
        row = dict(source)
        row.update(
            {
                "valid": True,
                "invalid_reason": None,
                "unique_count": 3,
                "strict_not_exact_pass": True,
                "weak_floor_pass": (
                    True if row["weak_floor_applicable"] else None
                ),
                "cell_verdict": "PASS_STRICT_INTERIOR",
            }
        )
        for name, value in values.items():
            row[name] = value
            row[name + "_float_hex"] = float(value).hex().lower()
        result.append(row)
    return tuple(result)


def synthetic_valid_distribution_records(
    diagnostic: str,
) -> tuple[dict[str, object], ...]:
    records = subject.empty_distribution_records(
        diagnostic, reason="STRUCTURALLY_NOT_APPLICABLE"
    )
    result: list[dict[str, object]] = []
    for source in records:
        row = dict(source)
        row["valid"] = True
        row["invalid_reason"] = None
        row["verdict"] = "PASS_STRICT_INTERIOR"
        if row["record_kind"] == "MONTHLY_IQR":
            for name in ("iqr_2022", "iqr_2023", "iqr_ratio"):
                row[name] = 1.0
                row[name + "_float_hex"] = float(1.0).hex().lower()
        else:
            edges = [float(index) / 10.0 for index in range(1, 10)]
            probabilities = [0.1] * 10
            row.update(
                {
                    "reference_internal_edges": edges,
                    "reference_internal_edges_float_hex": [
                        value.hex().lower() for value in edges
                    ],
                    "final_bin_count": 10,
                    "reference_counts": [1] * 10,
                    "comparison_counts": [1] * 10,
                    "reference_probabilities": probabilities,
                    "reference_probabilities_float_hex": [
                        value.hex().lower() for value in probabilities
                    ],
                    "comparison_probabilities": probabilities,
                    "comparison_probabilities_float_hex": [
                        value.hex().lower() for value in probabilities
                    ],
                    "psi": 0.0,
                    "psi_float_hex": float(0.0).hex().lower(),
                }
            )
        result.append(row)
    return tuple(result)


def completed_raw_evaluations() -> dict[str, subject.ResponseEvaluation]:
    actual = np.linspace(0.0, 1.0, subject.EXPECTED_SITE_ROWS, dtype=np.float64)
    prediction = np.ascontiguousarray(actual + np.float64(0.1))
    pooled = subject.pooled_nrmse_denominator(actual)
    metric_result = subject.regression_metrics(
        actual, prediction, pooled.denominator
    ).require_valid()
    result: dict[str, subject.ResponseEvaluation] = {}
    for response in subject.RESPONSES:
        component_result = raw_result(response, "PASS_NONREDUNDANT_MARGIN")
        duplicate_records = tuple(
            [
                subject._duplicate_metric_record(
                    record_kind="POOLED_ESTIMATOR",
                    response=response,
                    estimator=estimator_name,
                    predictor=None,
                    metric=metric_result,
                    pooled=pooled,
                    independent_selected=estimator_name == "RIDGE_PIPELINE",
                    parent_max_r2_witness=estimator_name == "EXTRA_TREES",
                    parent_redundancy_boolean=False,
                    component_classification="PASS_NONREDUNDANT_MARGIN",
                )
                for estimator_name in ("RIDGE_PIPELINE", "EXTRA_TREES")
            ]
            + [
                subject._duplicate_metric_record(
                    record_kind="AFFINE_PREDICTOR",
                    response=response,
                    estimator=None,
                    predictor=predictor,
                    metric=metric_result,
                    pooled=pooled,
                    pearson=0.0,
                    spearman=0.0,
                    affine_oof_nrmse=metric_result.nrmse,
                    component_classification="PASS_NONREDUNDANT_MARGIN",
                )
                for predictor in subject.PREDICTORS
            ]
        )
        result[response] = subject.ResponseEvaluation(
            response=response,
            actual=actual,
            pooled=pooled,
            ridge_oof=prediction,
            extra_trees_oof=prediction,
            independent_oof=prediction,
            independent_estimator="RIDGE_PIPELINE",
            raw_result=component_result,
            duplicate_metric_records=duplicate_records,
            stability_records=synthetic_valid_stability_records(
                response,
                diagnostic_kind="RAW_RESPONSE",
                denominator=pooled.denominator,
            ),
            distribution_records=synthetic_valid_distribution_records(response),
            threshold_witnesses=(),
            tied_witnesses=(),
            completed_fit_records=synthetic_completed_fit_records(response),
            skip_reason=None,
        )
    return result


def completed_derived_evaluation(
    raw_evaluations: dict[str, subject.ResponseEvaluation],
) -> subject.DerivedEvaluation:
    raw_rows = {
        response: dict(evaluation.raw_result)
        for response, evaluation in raw_evaluations.items()
    }
    endpoints = tuple(
        endpoint_result(
            diagnostic, raw_rows, "PASS_ENDPOINT_NONREDUNDANT_MARGIN"
        )
        for diagnostic in subject.ENDPOINT_VETO_DIAGNOSTICS
    )
    pooled_records: list[dict[str, object]] = []
    for endpoint in endpoints:
        pooled_records.append(
            {
                "diagnostic": endpoint["diagnostic"],
                "rows": subject.EXPECTED_SITE_ROWS,
                "valid": True,
                "invalid_reason": None,
                "source_component_estimator_binding": endpoint[
                    "source_component_estimator_binding"
                ],
                "sse": 1.0,
                "sse_float_hex": float(1.0).hex().lower(),
                "rmse": 0.1,
                "rmse_float_hex": float(0.1).hex().lower(),
                "r2": 0.9,
                "r2_float_hex": float(0.9).hex().lower(),
                "nrmse": 0.1,
                "nrmse_float_hex": float(0.1).hex().lower(),
                "actual_q05": 0.05,
                "actual_q05_float_hex": float(0.05).hex().lower(),
                "actual_q95": 0.95,
                "actual_q95_float_hex": float(0.95).hex().lower(),
                "nrmse_denominator": 0.9,
                "nrmse_denominator_float_hex": float(0.9).hex().lower(),
                "r2_boundary_equal_0_9925": False,
                "nrmse_boundary_equal_0_025": False,
                "strict_stable_margin_pass": True,
                "endpoint_pooled_verdict": (
                    "PASS_ENDPOINT_NONREDUNDANT_MARGIN"
                ),
            }
        )
    stability = tuple(
        item
        for diagnostic in subject.ENDPOINT_VETO_DIAGNOSTICS
        for item in synthetic_valid_stability_records(
            diagnostic,
            diagnostic_kind="ENDPOINT_VETO",
            denominator=np.float64(0.9),
        )
    )
    distribution = tuple(
        item
        for diagnostic in subject.ENDPOINT_VETO_DIAGNOSTICS
        for item in synthetic_valid_distribution_records(diagnostic)
    )
    return subject.DerivedEvaluation(
        site_actual={},
        site_prediction={},
        group_metadata=None,
        group_actual={},
        group_prediction={},
        endpoint_results=endpoints,
        endpoint_pooled_records=tuple(pooled_records),
        stability_records=stability,
        distribution_records=distribution,
        threshold_witnesses=(),
    )


def synthetic_sample_schedule() -> dict[str, np.ndarray]:
    operating_days = pd.DatetimeIndex(
        [
            pd.Timestamp(year=year, month=month, day=day)
            for year in (2022, 2023)
            for month in range(1, 13)
            for day in (5, 20)
        ]
    )
    forecast_local = pd.DatetimeIndex(
        [
            operating_day + pd.Timedelta(hours=hour)
            for operating_day in operating_days
            for hour in range(1, 25)
        ]
    )
    forecast_utc = forecast_local.tz_localize(
        "Asia/Seoul"
    ).tz_convert("UTC").as_unit("ns")
    run_utc = pd.DatetimeIndex(
        [
            operating_day - pd.Timedelta(days=2) + pd.Timedelta(hours=12)
            for operating_day in operating_days
        ]
    ).tz_localize("UTC")
    available_local = pd.DatetimeIndex(
        [
            operating_day - pd.Timedelta(days=1) + pd.Timedelta(hours=13)
            for operating_day in operating_days
        ]
    )
    return {
        "valid_time_utc": forecast_utc.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ).to_numpy(),
        "valid_time_ns": forecast_utc.asi8,
        "target_operating_day_kst": np.repeat(
            operating_days.strftime("%Y-%m-%d").to_numpy(), 24
        ),
        "forecast_kst_dtm": forecast_local.strftime(
            "%Y-%m-%d %H:%M:%S"
        ).to_numpy(),
        "data_available_kst_dtm": np.repeat(
            available_local.strftime("%Y-%m-%d %H:%M:%S").to_numpy(), 24
        ),
        "run_init_utc": np.repeat(
            run_utc.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(), 24
        ),
        "forecast_hour": np.tile(
            np.arange(28, 52, dtype=np.int64), len(operating_days)
        ),
    }


def synthetic_output_prepared(
    tmp_path: subject.Path,
) -> tuple[subject.PreparedExecution, pd.DataFrame]:
    schedule = synthetic_sample_schedule()
    timestamp_ns = schedule["valid_time_ns"]
    operating_days = schedule["target_operating_day_kst"]
    groups = np.asarray(
        ["kpx_group_1"] * 6
        + ["kpx_group_2"] * 6
        + ["kpx_group_3"] * 5,
        dtype=object,
    )
    site = pd.DataFrame(
        {
            "valid_time_utc": np.repeat(timestamp_ns, 17),
            "target_operating_day_kst__external": np.repeat(
                operating_days, 17
            ),
            "forecast_hour": np.repeat(
                schedule["forecast_hour"], 17
            ).astype(np.int16),
            "site_id": np.tile(
                np.arange(1, 18, dtype=np.int16),
                subject.EXPECTED_TIMESTAMPS,
            ),
            "group__external": np.tile(
                groups, subject.EXPECTED_TIMESTAMPS
            ),
            "capacity_mw__external": np.ones(
                subject.EXPECTED_SITE_ROWS, dtype=np.float64
            ),
        }
    )
    group_metadata = pd.DataFrame(
        {
            "valid_time_utc": np.repeat(timestamp_ns, 3),
            "group": np.tile(
                np.asarray(
                    ["kpx_group_1", "kpx_group_2", "kpx_group_3"],
                    dtype=object,
                ),
                subject.EXPECTED_TIMESTAMPS,
            ),
            "capacity_mw": np.tile(
                np.asarray([6.0, 6.0, 5.0], dtype=np.float64),
                subject.EXPECTED_TIMESTAMPS,
            ),
        }
    )
    fold_ordinals = np.repeat(
        np.repeat(np.arange(1, 5, dtype=np.int8), 288), 17
    )
    authorization = subject.PredataAuthorization(
        valid_control(), arrow_handoff_started=True
    )
    return (
        subject.PreparedExecution(
            artifact_root=tmp_path,
            attempt_id="target_free_duplicate_v5__20260811T123456123456Z",
                amendment={},
                amendment_v4={},
                amendment_v5={},
                v3_failure_incident={},
                bound_paths={},
            authorization=authorization,
            joined=subject.JoinedInputs(
                site=site,
                group=pd.DataFrame(),
                predictor_matrix_float64=np.empty((0, 0), dtype=np.float64),
                fold_ordinals=fold_ordinals,
            ),
            runtime_evidence={},
            alias_identities=(),
        ),
        group_metadata,
    )


def alias_closure_fixture(
    tmp_path: subject.Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    semantic_tamper: bool = False,
    provenance_rows: int = 10_368,
    missing_provenance_field: bool = False,
) -> tuple[
    subject.PredataAuthorization,
    tuple[subject.FileIdentity, ...],
]:
    def make_identity(
        relative_path: str, payload: bytes, *, absolute: bool = False
    ) -> dict[str, object]:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return {
            "path": path.resolve().as_posix() if absolute else relative_path,
            "size_bytes": len(payload),
            "sha256": subject.hashlib.sha256(payload).hexdigest(),
        }

    identity_names = (
        "access_ledger",
        "day_summary",
        "field_range_census",
        "field_summary",
        "object_census",
        "parent_preregister_manifest",
        "raw_download_plan",
        "summary",
    )
    nested = {
        name: make_identity(f"metadata/{name}.bin", name.encode("ascii"))
        for name in identity_names
    }
    nested["reproduction_code"] = make_identity(
        "synthetic_code.py", b"# synthetic code\n", absolute=True
    )
    nested["test_code"] = make_identity(
        "synthetic_test.py", b"# synthetic test\n", absolute=True
    )
    checkpoints = [
        make_identity(
            f"metadata/progress_{index:02d}.json",
            ("checkpoint-" + str(index)).encode("ascii"),
        )
        for index in range(13)
    ]
    census = {
        "access_ledger": nested["access_ledger"],
        "artifact_type": "NOAA_GFS_MULTISEASON_CENSUS_MANIFEST",
        "created_utc": "2026-08-10T13:47:10.975928Z",
        "day_summary": nested["day_summary"],
        "field_range_census": nested["field_range_census"],
        "field_summary": nested["field_summary"],
        "labels_read": semantic_tamper,
        "models_fit": 0,
        "object_census": nested["object_census"],
        "parent_preregister_manifest": nested["parent_preregister_manifest"],
        "progress_checkpoints": checkpoints,
        "raw_download_plan": nested["raw_download_plan"],
        "raw_downloaded_bytes": 0,
        "raw_network_requests": 0,
        "reproduction_code": nested["reproduction_code"],
        "schema_version": 1,
        "submission_csv_created": False,
        "summary": nested["summary"],
        "test_code": nested["test_code"],
    }
    census_payload = subject.strict_json_dumps(census, pretty=True)
    provenance_payload = b"synthetic-provenance-footer"
    census_source_rel = "source/FIELD_CENSUS_SOURCE.json"
    provenance_source_rel = "source/PROVENANCE_SOURCE.parquet"
    census_source = tmp_path / census_source_rel
    provenance_source = tmp_path / provenance_source_rel
    census_source.parent.mkdir(parents=True, exist_ok=True)
    census_source.write_bytes(census_payload)
    provenance_source.write_bytes(provenance_payload)
    output_root = tmp_path / subject.OUTPUT_ROOT_RELATIVE
    output_root.mkdir(parents=True, exist_ok=True)
    census_final_rel = (
        subject.OUTPUT_ROOT_RELATIVE + "/FIELD_CENSUS_LOCK.json"
    )
    provenance_final_rel = (
        subject.OUTPUT_ROOT_RELATIVE + "/PROVENANCE_LEDGER.parquet"
    )
    (tmp_path / census_final_rel).write_bytes(census_payload)
    (tmp_path / provenance_final_rel).write_bytes(provenance_payload)
    census_identity = subject.FileIdentity(
        census_source_rel,
        len(census_payload),
        subject.hashlib.sha256(census_payload).hexdigest(),
    )
    provenance_identity = subject.FileIdentity(
        provenance_source_rel,
        len(provenance_payload),
        subject.hashlib.sha256(provenance_payload).hexdigest(),
    )
    specs = (
        subject.AliasSpec(
            census_identity,
            subject.OUTPUT_ROOT_RELATIVE + "/.stage-census",
            census_final_rel,
        ),
        subject.AliasSpec(
            provenance_identity,
            subject.OUTPUT_ROOT_RELATIVE + "/.stage-provenance",
            provenance_final_rel,
        ),
    )
    monkeypatch.setattr(subject, "ALIAS_SPECS", specs)
    names = list(subject.PROVENANCE_REQUIRED_FIELDS)
    if missing_provenance_field:
        names.pop()
    fake_footer = type(
        "Footer",
        (),
        {
            "schema_arrow": type("Schema", (), {"names": names})(),
            "metadata": type(
                "Metadata", (), {"num_rows": provenance_rows}
            )(),
            "close": lambda self: None,
        },
    )()
    monkeypatch.setattr(
        subject,
        "guarded_alias_parquet_file",
        lambda *_args, **_kwargs: fake_footer,
    )
    authorization = subject.PredataAuthorization(
        valid_control(aliases_pass=False),
        artifact_root=tmp_path,
        alias_files_published=True,
    )
    final_identities = (
        subject.FileIdentity(
            census_final_rel,
            census_identity.size_bytes,
            census_identity.sha256,
        ),
        subject.FileIdentity(
            provenance_final_rel,
            provenance_identity.size_bytes,
            provenance_identity.sha256,
        ),
    )
    return authorization, final_identities


def zero_threadpool_evidence() -> dict[str, object]:
    observations = [
        {
            "name": name,
            "path": subject.NATIVE_LIBRARY_IDENTITIES[name].path,
            **dict(subject.THREADPOOL_METADATA_EXACT[name]),
            "num_threads": 1,
        }
        for name in subject.NATIVE_LIBRARY_IDENTITIES
    ]
    before = {
        "event_ordinal": 1,
        "phase": "BEFORE_EXPLICIT_CONTEXT",
        "label": "synthetic-before",
        "threadpools": observations,
    }
    return {
        "capture_phases": list(subject.THREADPOOL_PHASES),
        "capture_counts": {
            "BEFORE_EXPLICIT_CONTEXT": 1,
            "INSIDE_CONTEXT": 0,
            "AFTER_CONTEXT": 0,
        },
        "event_count": 1,
        "event_chain_sha256": "0" * 64,
        "first_observation_by_phase": {
            "BEFORE_EXPLICIT_CONTEXT": before,
            "INSIDE_CONTEXT": None,
            "AFTER_CONTEXT": None,
        },
        "last_observation_by_phase": {
            "BEFORE_EXPLICIT_CONTEXT": before,
            "INSIDE_CONTEXT": None,
            "AFTER_CONTEXT": None,
        },
    }


def test_contract_constants_and_planned_fit_slot_counts() -> None:
    assert len(subject.RESPONSES) == 9
    assert len(subject.PREDICTORS) == 35
    units = subject.planned_fit_units()
    assert len(units) == 1332
    assert sum(unit.unit_kind == "PRIMARY_MODEL" for unit in units) == 72
    assert sum(unit.unit_kind == "ANALYTIC_AFFINE" for unit in units) == 1260
    assert units[0].planned_unit_slot_id == "PMU__01__1__1"
    assert units[71].planned_unit_slot_id == "PMU__09__4__2"
    assert units[72].planned_unit_slot_id == "AAU__01__01__1"
    assert units[-1].planned_unit_slot_id == "AAU__09__35__4"
    assert [unit.decision_fit_ordinal for unit in units] == list(range(1, 1333))
    assert len(subject.COMPLETE_ARTIFACT_BASENAMES) == 11
    assert subject.COMPLETE_ARTIFACT_BASENAMES[-1] == (
        "TRACK_A_TARGET_FREE_RUN_MANIFEST.json"
    )
    assert subject.planned_oof_slot_id(9, 2, 19584) == (
        "OOF__09__2__19584"
    )
    assert subject.planned_derived_slot_id(18, "GROUP", 3456) == (
        "DWD__18__G__03456"
    )


def test_conditional_ledger_keeps_all_1332_slots_with_zero_calls() -> None:
    reasons = {
        response: (
            "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3"
        )
        for response in subject.RESPONSES
    }
    ledger = subject.skipped_fit_ledger(reasons)
    assert len(ledger) == 1332
    assert tuple(ledger[0]) == subject.PRIMARY_LEDGER_KEYS
    assert tuple(ledger[72]) == subject.AFFINE_LEDGER_KEYS
    assert {row["unit_status"] for row in ledger} == {
        "SKIPPED_PREMODEL_CLEAR_VETO"
    }
    assert sum(row.get("pipeline_fit_calls", 0) for row in ledger) == 0
    assert sum(row.get("extra_trees_fit_calls", 0) for row in ledger) == 0
    assert all(row["fit_completed"] is False for row in ledger)


def test_premodel_constant_skip_is_local_but_predictor_nonfinite_is_integrity() -> None:
    assert subject.premodel_response_skip_reason(np.zeros(19584)) == (
        "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3"
    )
    with pytest.raises(subject.IntegrityError, match="non-finite"):
        subject.premodel_response_skip_reason(np.array([0.0, 1.0, np.nan]))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_nonfinite_constants(constant: str) -> None:
    with pytest.raises(subject.StrictJSONError, match="forbidden"):
        subject.strict_json_loads('{"x":' + constant + "}")


def test_strict_json_rejects_recursive_duplicate_and_nonfinite_dump() -> None:
    with pytest.raises(subject.StrictJSONError, match="duplicate"):
        subject.strict_json_loads('{"outer":{"x":1,"x":2}}')
    with pytest.raises(subject.StrictJSONError, match="non-finite"):
        subject.strict_json_dumps({"x": float("nan")}, pretty=True)
    assert subject.strict_json_dumps({"é": 1}, pretty=True).endswith(b"\n")
    assert b"\\u00e9" in subject.strict_json_dumps({"é": 1}, pretty=True)


@pytest.mark.parametrize(
    "variant", ("compact", "spacing", "missing_terminal_lf", "raw_non_ascii")
)
def test_control_files_require_exact_pretty_json_bytes(
    tmp_path: subject.Path, variant: str
) -> None:
    relative = "prereg/synthetic_control.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    payload = {"message": "풍력", "nested": {"b": 2, "a": 1}}
    pretty = subject.strict_json_dumps(payload, pretty=True)
    path.write_bytes(pretty)
    observed, _, _ = subject._load_control(tmp_path, relative)
    assert observed == payload

    if variant == "compact":
        drift = subject.strict_json_dumps(payload, pretty=False)
    elif variant == "spacing":
        drift = pretty.replace(b'  "message"', b'   "message"', 1)
    elif variant == "missing_terminal_lf":
        drift = pretty[:-1]
    else:
        drift = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    assert drift != pretty
    path.write_bytes(drift)
    with pytest.raises(
        subject.PredataAuthorizationError,
        match="deterministic pretty JSON serialization",
    ):
        subject._load_control(tmp_path, relative)


def test_v4_csv_boolean_and_active_boundary_flag_grammar_is_exact() -> None:
    columns = ("valid", "boundary_equality_flags")
    active = ";".join(
        (
            "AFFINE_ABS_PEARSON_0.9999",
            "CELL_NRMSE_0.02",
            "YEAR_PSI_0.5",
        )
    )
    payload = subject.serialize_csv_rows(
        (
            {"valid": True, "boundary_equality_flags": ""},
            {"valid": False, "boundary_equality_flags": active},
        ),
        columns,
    )
    assert payload.decode("utf-8").splitlines() == [
        "valid,boundary_equality_flags",
        "TRUE,",
        f"FALSE,{active}",
    ]
    with pytest.raises(subject.IntegrityError, match="boolean cell"):
        subject.serialize_csv_rows(
            ({"valid": "true", "boundary_equality_flags": ""},), columns
        )
    for invalid in (
        "UNKNOWN_CUTOFF",
        "CELL_NRMSE_0.02;AFFINE_ABS_PEARSON_0.9999",
        "CELL_NRMSE_0.02;CELL_NRMSE_0.02",
        "TRUE;FALSE",
    ):
        with pytest.raises(subject.IntegrityError, match="boundary equality"):
            subject.serialize_csv_rows(
                ({"valid": True, "boundary_equality_flags": invalid},),
                columns,
            )


def test_published_v4_identity_and_exact_two_pointer_overlay(
    tmp_path: subject.Path,
) -> None:
    repository_root = subject.Path(subject.__file__).resolve().parents[1]
    source_root = (
        repository_root
        / "artifacts"
        / "baram2026_ncei_scada_longrun_20260810_v2"
    )
    (tmp_path / "prereg").mkdir()
    for relative in (
        subject.AMENDMENT_RELATIVE_PATH,
        subject.AMENDMENT_V4_RELATIVE_PATH,
    ):
        (tmp_path / relative).write_bytes((source_root / relative).read_bytes())
    amendment = subject.validate_amendment(tmp_path)
    amendment_v4 = subject.validate_amendment_v4(tmp_path, amendment)
    effective = subject.validate_v3_v4_overlay(amendment, amendment_v4)
    assert effective["parquet"] == subject.PARQUET_SERIALIZATION_LITERAL_V4
    assert (
        effective["csv_boundary_flags"]
        == subject.CSV_BOUNDARY_SERIALIZATION_LITERAL_V4
    )
    changed = {
        key
        for key, value in effective.items()
        if value
        != amendment["output_contract"]["output_serialization_exact"][key]
    }
    assert changed == {"parquet", "csv_boundary_flags"}
    assert len(subject.PARQUET_WRITE_TABLE_KWARGS_EXACT) == 12
    assert subject.PARQUET_WRITE_TABLE_KWARGS_EXACT["coerce_timestamps"] is None
    drift = copy.deepcopy(amendment_v4)
    drift["correction_scope"]["correction_count_exact"] = 3
    with pytest.raises(subject.IdentityError, match="correction scope"):
        subject.validate_v3_v4_overlay(amendment, drift)


def test_published_v5_standalone_identity_lineage_and_generation_contract(
    tmp_path: subject.Path,
) -> None:
    source_root = subject.ARTIFACT_ROOT_DEFAULT
    for relative in (
        subject.AMENDMENT_RELATIVE_PATH,
        subject.AMENDMENT_V4_RELATIVE_PATH,
        subject.AMENDMENT_V5_RELATIVE_PATH,
        subject.V3_FAILURE_INCIDENT_RELATIVE_PATH,
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / relative).read_bytes())
    amendment = subject.validate_amendment(tmp_path)
    amendment_v4 = subject.validate_amendment_v4(tmp_path, amendment)
    incident = subject.validate_v3_failure_incident(tmp_path)
    amendment_v5 = subject.validate_amendment_v5(
        tmp_path, amendment, amendment_v4, incident
    )
    assert len(amendment_v5) == 24
    assert amendment_v5["control_contract"]["control_schema_version_exact"] == 5
    assert amendment_v5["code_and_test_paths_exact"] == dict(
        subject.CODE_IDENTITY_FILES
    )
    assert amendment_v5["control_contract"]["output_namespace_exact"] == dict(
        subject.CONTROL_OUTPUT_NAMESPACE
    )
    assert amendment_v5["execution_budget"] == dict(subject.EXECUTION_BUDGET)
    assert amendment_v5["output_contract"][
        "final_publication_order_positive_exact"
    ] == list(subject.RUNNER_ARTIFACT_BASENAMES) + list(
        subject.SEALER_ARTIFACT_BASENAMES
    )
    assert amendment_v5["output_contract"][
        "final_publication_order_negative_exact"
    ] == amendment_v5["output_contract"][
        "final_publication_order_positive_exact"
    ]
    broken = tmp_path / subject.AMENDMENT_V5_RELATIVE_PATH
    broken.write_bytes(broken.read_bytes() + b" ")
    with pytest.raises(subject.IdentityError):
        subject.validate_amendment_v5(
            tmp_path, amendment, amendment_v4, incident
        )


def test_v4_common_control_top_keysets_and_identity_are_mandatory() -> None:
    assert {
        role: len(keys) for role, keys in subject.CONTROL_TOP_KEYS.items()
    } == {
        "CODE_SEAL": 13,
        "AUTHORIZATION": 17,
        "REVIEW": 16,
        "GO": 17,
        "POSTRUN": 18,
    }
    assert all(
        keys.count("amendment_v4") == 1
        for keys in subject.CONTROL_TOP_KEYS.values()
    )
    code_identities = {
        role: {"path": f"{role}.py", "size_bytes": 1, "sha256": "0" * 64}
        for role in subject.CODE_IDENTITY_KEYS
    }
    record = {
        "schema_version": 5,
        **dict(subject.CONTROL_LITERALS["CODE_SEAL"]),
        "created_utc": "2026-08-11T16:00:00.000000Z",
        "amendment": subject._expected_amendment_identity(),
        "amendment_v4": subject._expected_amendment_v4_identity(),
        "amendment_v5": subject._expected_amendment_v5_identity(),
        "code_identities": code_identities,
        "test_evidence": {},
        "output_namespace": dict(subject.CONTROL_OUTPUT_NAMESPACE),
        "authority_scope": dict(subject.DOCUMENTARY_AUTHORITY_SCOPE),
        "required_next_controls": dict(subject.REQUIRED_NEXT_CONTROLS),
    }
    subject._validate_control_envelope("CODE_SEAL", record)
    missing = dict(record)
    del missing["amendment_v5"]
    with pytest.raises(subject.PredataAuthorizationError, match="key set"):
        subject._validate_control_envelope("CODE_SEAL", missing)
    wrong = copy.deepcopy(record)
    wrong["amendment_v5"]["sha256"] = "1" * 64
    with pytest.raises(subject.PredataAuthorizationError, match="amendment_v5"):
        subject._validate_control_envelope("CODE_SEAL", wrong)

    numeric_schema_alias = copy.deepcopy(record)
    numeric_schema_alias["schema_version"] = 3.0
    with pytest.raises(subject.PredataAuthorizationError, match="schema_version"):
        subject._validate_control_envelope("CODE_SEAL", numeric_schema_alias)

    numeric_identity_alias = copy.deepcopy(record)
    numeric_identity_alias["amendment"]["size_bytes"] = float(
        numeric_identity_alias["amendment"]["size_bytes"]
    )
    with pytest.raises(subject.PredataAuthorizationError, match="amendment identity"):
        subject._validate_control_envelope("CODE_SEAL", numeric_identity_alias)

    true_as_one = copy.deepcopy(record)
    true_as_one["authority_scope"]["documentary_only"] = 1
    with pytest.raises(subject.PredataAuthorizationError, match="authority scope"):
        subject._validate_control_envelope("CODE_SEAL", true_as_one)

    false_as_zero = copy.deepcopy(record)
    false_as_zero["authority_scope"]["execution_authorized"] = 0
    with pytest.raises(subject.PredataAuthorizationError, match="authority scope"):
        subject._validate_control_envelope("CODE_SEAL", false_as_zero)

    namespace_true_as_one = copy.deepcopy(record)
    namespace_true_as_one["output_namespace"]["single_attempt_only"] = 1
    with pytest.raises(subject.PredataAuthorizationError, match="output namespace"):
        subject._validate_control_envelope("CODE_SEAL", namespace_true_as_one)
    assert subject.CODE_CONFIG_RUNTIME_IDENTITY_KEYS == (
        "amendment",
        "amendment_v4",
        "amendment_v5",
        "code_identities",
        "runtime_identity",
    )


def test_physical_code_identities_are_absolute_and_never_coerce_json_scalars(
    tmp_path: subject.Path,
) -> None:
    repository = tmp_path / "repo"
    artifact_root = repository / "artifacts" / "synthetic"
    artifact_root.mkdir(parents=True)
    identities: dict[str, dict[str, object]] = {}
    for ordinal, role in enumerate(subject.CODE_IDENTITY_KEYS, start=1):
        path = repository / subject.CODE_IDENTITY_FILES[role]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"role-{ordinal}\n".encode("ascii"))
        size, digest = subject._sha256_and_size(path)
        identities[role] = {
            "path": path.resolve(strict=True).as_posix(),
            "size_bytes": size,
            "sha256": digest,
        }

    subject._validate_physical_code_identities(artifact_root, identities)

    float_size = copy.deepcopy(identities)
    float_size["runner"]["size_bytes"] = float(
        float_size["runner"]["size_bytes"]
    )
    with pytest.raises(subject.PredataAuthorizationError, match="scalar type"):
        subject._validate_physical_code_identities(artifact_root, float_size)

    integer_path = copy.deepcopy(identities)
    integer_path["runner"]["path"] = 1
    with pytest.raises(subject.PredataAuthorizationError, match="scalar type"):
        subject._validate_physical_code_identities(artifact_root, integer_path)

    non_string_sha = copy.deepcopy(identities)
    non_string_sha["runner"]["sha256"] = 0
    with pytest.raises(subject.PredataAuthorizationError, match="scalar type"):
        subject._validate_physical_code_identities(artifact_root, non_string_sha)


def test_predata_guard_blocks_values_while_allowing_only_final_alias_footer(
    tmp_path: subject.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = subject.PredataAuthorization(
        valid_control(independent_go_pass=False)
    )
    for function in (
        subject.guarded_parquet_file,
        subject.guarded_read_table,
        subject.guarded_pandas_read_parquet,
    ):
        with pytest.raises(subject.PredataAuthorizationError):
            function(authorization, subject.Path("forbidden.parquet"))
    output_root = tmp_path / subject.OUTPUT_ROOT_RELATIVE
    output_root.mkdir(parents=True)
    provenance = output_root / "PROVENANCE_LEDGER.parquet"
    provenance.write_bytes(b"synthetic-footer")
    metadata_only = subject.PredataAuthorization(
        valid_control(aliases_pass=False),
        artifact_root=tmp_path,
        alias_files_published=True,
    )
    import pyarrow.parquet as pq

    marker = object()
    monkeypatch.setattr(pq, "ParquetFile", lambda *_args, **_kwargs: marker)
    assert subject.guarded_alias_parquet_file(
        metadata_only, provenance
    ) is marker
    for function in (
        subject.guarded_parquet_file,
        subject.guarded_read_table,
        subject.guarded_pandas_read_parquet,
        subject.guarded_bound_value_open,
    ):
        with pytest.raises(subject.PredataAuthorizationError):
            function(metadata_only, provenance)
    metadata_only.alias_metadata_validated = True
    metadata_only.validation = subject.dataclasses.replace(
        metadata_only.validation, aliases_pass=True
    )
    metadata_only.authorize_arrow_handoff()
    assert metadata_only.arrow_handoff_started is True


def test_parquet_footer_resources_close_on_success_and_validation_failure(
    tmp_path: subject.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow as pa

    closed: list[str] = []

    class SyntheticSchema:
        names = ("x",)

        def __iter__(self):
            return iter((type("Field", (), {"type": pa.float64()})(),))

        def equals(self, other: object, *, check_metadata: bool) -> bool:
            return check_metadata and isinstance(other, SyntheticSchema)

    class SyntheticParquetFile:
        schema_arrow = SyntheticSchema()
        metadata = type("Metadata", (), {"num_rows": 1})()

        def __init__(self, marker: str) -> None:
            self.marker = marker

        def close(self) -> None:
            closed.append(self.marker)

    resources = iter(
        (
            SyntheticParquetFile("success"),
            SyntheticParquetFile("failure"),
            SyntheticParquetFile("schema-success"),
            SyntheticParquetFile("schema-failure"),
        )
    )
    monkeypatch.setattr(
        subject,
        "guarded_parquet_file",
        lambda *_args, **_kwargs: next(resources),
    )
    authorization = subject.PredataAuthorization(valid_control())
    subject._validate_parquet_names_types_rows(
        authorization,
        tmp_path / "synthetic.parquet",
        expected_columns=("x",),
        expected_type_names=("float64",),
        expected_rows=1,
    )
    with pytest.raises(subject.IntegrityError, match="row count"):
        subject._validate_parquet_names_types_rows(
            authorization,
            tmp_path / "synthetic.parquet",
            expected_columns=("x",),
            expected_type_names=("float64",),
            expected_rows=2,
        )
    handoff = subject.PredataAuthorization(
        valid_control(), arrow_handoff_started=True
    )
    observed = subject.guarded_parquet_schema(
        handoff,
        tmp_path / "synthetic.parquet",
        expected_arrow_schema=SyntheticSchema(),
        expected_rows=1,
    )
    assert isinstance(observed, SyntheticSchema)
    with pytest.raises(subject.IntegrityError, match="row count"):
        subject.guarded_parquet_schema(
            handoff,
            tmp_path / "synthetic.parquet",
            expected_arrow_schema=SyntheticSchema(),
            expected_rows=2,
        )
    assert closed == [
        "success",
        "failure",
        "schema-success",
        "schema-failure",
    ]


def test_synthetic_parquet_writer_freezes_physical_and_logical_contract(
    tmp_path: subject.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        (
            pa.field(
                "valid_time_utc", pa.timestamp("ns", tz="UTC"), nullable=False
            ),
            pa.field(
                "target_operating_day_kst", pa.date32(), nullable=False
            ),
            pa.field("ordinal", pa.int16(), nullable=False),
            pa.field("value", pa.float64(), nullable=True),
        )
    )
    arrays = (
        pa.array(
            [
                1_650_000_000_000_000_123,
                1_650_000_000_000_000_124,
                1_650_000_000_000_000_125,
            ],
            type=pa.timestamp("ns", tz="UTC"),
        ),
        pa.array(
            [dt.date(2022, 1, 5), dt.date(2022, 1, 5), dt.date(2022, 1, 5)],
            type=pa.date32(),
        ),
        pa.array([1, 2, 3], type=pa.int16()),
        pa.array([0.0, None, 1.0], type=pa.float64()),
    )
    original_write_table = pq.write_table
    observed_kwargs: list[dict[str, object]] = []

    def recording_write_table(
        table: object, where: object, **kwargs: object
    ) -> None:
        observed_kwargs.append(dict(kwargs))
        original_write_table(table, where, **kwargs)

    monkeypatch.setattr(pq, "write_table", recording_write_table)
    path = tmp_path / "synthetic.parquet"
    identity = subject.write_parquet_exclusive(
        path, schema=schema, arrays=arrays, row_count=3
    )
    second_path = tmp_path / "synthetic_repeat.parquet"
    second_identity = subject.write_parquet_exclusive(
        second_path, schema=schema, arrays=arrays, row_count=3
    )
    assert observed_kwargs == [
        dict(subject.PARQUET_WRITE_TABLE_KWARGS_EXACT),
        dict(subject.PARQUET_WRITE_TABLE_KWARGS_EXACT),
    ]
    assert len(observed_kwargs[0]) == 12
    assert observed_kwargs[0]["coerce_timestamps"] is None
    assert path.read_bytes() == second_path.read_bytes()
    assert (identity.size_bytes, identity.sha256) == (
        second_identity.size_bytes,
        second_identity.sha256,
    )
    assert identity.format == "PARQUET"
    assert identity.row_count == 3
    assert (identity.size_bytes, identity.sha256) == subject._sha256_and_size(
        path
    )
    parquet_file = pq.ParquetFile(path)
    try:
        assert parquet_file.schema_arrow.equals(schema, check_metadata=True)
        assert parquet_file.metadata.num_rows == 3
        assert parquet_file.metadata.num_row_groups == 1
        assert parquet_file.metadata.format_version == "2.6"
        assert b"pandas" not in (parquet_file.schema_arrow.metadata or {})
        roundtrip = parquet_file.read(use_threads=False)
        assert identity.logical_sha256 == subject.arrow_logical_sha256(
            roundtrip, schema
        )
        assert roundtrip["valid_time_utc"].cast(pa.int64()).to_pylist()[0] % 1000 == 123
        assert roundtrip.equals(
            pa.Table.from_arrays(list(arrays), schema=schema),
            check_metadata=True,
        )
        for column_index in range(len(schema)):
            column = parquet_file.metadata.row_group(0).column(column_index)
            assert column.compression == "ZSTD"
            assert "RLE_DICTIONARY" not in column.encodings
            assert column.statistics is not None
    finally:
        parquet_file.close()
    with pytest.raises(subject.OutputPublicationError, match="path exists"):
        subject.write_parquet_exclusive(
            path, schema=schema, arrays=arrays, row_count=3
        )
    table = pa.Table.from_arrays(list(arrays), schema=schema)
    invalid_ns_kwargs = dict(subject.PARQUET_WRITE_TABLE_KWARGS_EXACT)
    invalid_ns_kwargs["coerce_timestamps"] = "ns"
    with pytest.raises(ValueError, match="coerce_timestamps"):
        original_write_table(table, pa.BufferOutputStream(), **invalid_ns_kwargs)
    forced_us_kwargs = dict(subject.PARQUET_WRITE_TABLE_KWARGS_EXACT)
    forced_us_kwargs["coerce_timestamps"] = "us"
    with pytest.raises(pa.ArrowInvalid, match="would lose data"):
        original_write_table(table, pa.BufferOutputStream(), **forced_us_kwargs)
    wrong_schema = pa.schema(
        (
            pa.field("valid_time_utc", pa.timestamp("us"), nullable=False),
            pa.field("target_operating_day_kst", pa.date64(), nullable=False),
            pa.field("value", pa.float32(), nullable=False),
        )
    )
    wrong_path = tmp_path / "wrong_schema.parquet"
    with pytest.raises(subject.IntegrityError, match="timestamp/date type"):
        subject.write_parquet_exclusive(
            wrong_path,
            schema=wrong_schema,
            arrays=(
                pa.array([1], type=pa.timestamp("us")),
                pa.array([0], type=pa.date64()),
                pa.array([1.0], type=pa.float32()),
            ),
            row_count=1,
        )
    assert not subject._lexists(wrong_path)


def test_pyarrow22_writer_signature_and_unlisted_defaults_are_pinned() -> None:
    import pyarrow
    import pyarrow.parquet as pq

    assert pyarrow.__version__ == "22.0.0"
    subject.validate_pyarrow_write_table_signature(pq.write_table)

    def drifted(table: object, where: object, **kwargs: object) -> None:
        del table, where, kwargs

    with pytest.raises(
        subject.PredataAuthorizationError, match="signature keys"
    ):
        subject.validate_pyarrow_write_table_signature(drifted)


def test_full_oof_arrow_materialization_has_exact_slots_schema_and_selection(
    tmp_path: subject.Path,
) -> None:
    import pyarrow as pa

    prepared, _ = synthetic_output_prepared(tmp_path)
    raw = completed_raw_evaluations()
    wrong_folds = prepared.joined.fold_ordinals.copy()
    wrong_folds[0] = np.int8(2)
    with pytest.raises(subject.IntegrityError, match="target operating day"):
        subject.build_oof_arrow_table(
            subject.dataclasses.replace(
                prepared,
                joined=subject.dataclasses.replace(
                    prepared.joined, fold_ordinals=wrong_folds
                ),
            ),
            raw,
        )
    schema, arrays = subject.build_oof_arrow_table(prepared, raw)
    table = pa.Table.from_arrays(list(arrays), schema=schema)
    assert tuple(schema.names) == subject.OOF_COLUMNS
    assert table.num_rows == subject.EXPECTED_OOF_ROWS
    assert str(schema.field("valid_time_utc").type) == "timestamp[ns, tz=UTC]"
    assert str(schema.field("target_operating_day_kst").type) == "date32[day]"
    assert arrays[0][0].as_py() == "OOF__01__1__00001"
    assert arrays[0][-1].as_py() == "OOF__09__2__19584"
    assert arrays[3][0].as_py() == "2022_H1"
    selected = arrays[15].to_numpy(zero_copy_only=False)
    assert int(np.count_nonzero(selected)) == (
        len(subject.RESPONSES) * subject.EXPECTED_SITE_ROWS
    )
    del table, arrays, raw, prepared
    gc.collect()


def test_full_derived_arrow_materialization_has_exact_slots_and_scope_order(
    tmp_path: subject.Path,
) -> None:
    import pyarrow as pa

    prepared, group_metadata = synthetic_output_prepared(tmp_path)
    raw = completed_raw_evaluations()
    derived_base = completed_derived_evaluation(raw)
    site_base = np.linspace(
        0.0, 1.0, subject.EXPECTED_SITE_ROWS, dtype=np.float64
    )
    group_base = np.linspace(
        0.0, 1.0, subject.EXPECTED_GROUP_ROWS, dtype=np.float64
    )
    derived = subject.dataclasses.replace(
        derived_base,
        site_actual={
            diagnostic: site_base + np.float64(index)
            for index, diagnostic in enumerate(subject.DERIVED_DIAGNOSTICS)
        },
        site_prediction={
            diagnostic: site_base + np.float64(index) + np.float64(0.1)
            for index, diagnostic in enumerate(subject.DERIVED_DIAGNOSTICS)
        },
        group_metadata=group_metadata,
        group_actual={
            diagnostic: group_base + np.float64(index)
            for index, diagnostic in enumerate(subject.DERIVED_DIAGNOSTICS)
        },
        group_prediction={
            diagnostic: group_base + np.float64(index) + np.float64(0.1)
            for index, diagnostic in enumerate(subject.DERIVED_DIAGNOSTICS)
        },
    )
    schema, arrays = subject.build_derived_arrow_table(prepared, raw, derived)
    table = pa.Table.from_arrays(list(arrays), schema=schema)
    assert tuple(schema.names) == subject.DERIVED_COLUMNS
    assert table.num_rows == subject.EXPECTED_DERIVED_ROWS
    assert str(schema.field("valid_time_utc").type) == "timestamp[ns, tz=UTC]"
    assert str(schema.field("target_operating_day_kst").type) == "date32[day]"
    assert arrays[0][0].as_py() == "DWD__01__S__00001"
    assert arrays[0][-1].as_py() == "DWD__18__G__03456"
    scope = arrays[3].to_numpy(zero_copy_only=False)
    assert int(np.count_nonzero(scope == "SITE")) == (
        len(subject.DERIVED_DIAGNOSTICS) * subject.EXPECTED_SITE_ROWS
    )
    assert int(np.count_nonzero(scope == "GROUP")) == (
        len(subject.DERIVED_DIAGNOSTICS) * subject.EXPECTED_GROUP_ROWS
    )
    del table, arrays, raw, derived, derived_base, prepared, group_metadata
    gc.collect()


def test_fold_membership_is_fixed_and_outside_dates_fail() -> None:
    assert subject.fold_ordinal_for_operating_day(dt.date(2022, 1, 1)) == 1
    assert subject.fold_ordinal_for_operating_day(dt.date(2022, 12, 31)) == 2
    assert subject.fold_ordinal_for_operating_day(dt.date(2023, 6, 30)) == 3
    assert subject.fold_ordinal_for_operating_day(dt.date(2023, 12, 31)) == 4
    with pytest.raises(subject.IntegrityError):
        subject.fold_ordinal_for_operating_day(dt.date(2024, 1, 1))


def test_exact_join_normalizes_utc_and_rejects_duplicates_or_key_drift() -> None:
    left = pd.DataFrame(
        {
            "valid_time_utc": ["2022-01-01T00:00:00Z", "2022-01-01T01:00:00Z"],
            "site_id": ["A", "B"],
            "actual": [1.0, 2.0],
        }
    )
    right = pd.DataFrame(
        {
            "valid_time_utc": pd.to_datetime(
                ["2022-01-01T00:00:00Z", "2022-01-01T01:00:00Z"], utc=True
            ),
            "site_id": ["A", "B"],
            "predictor": [3.0, 4.0],
        }
    )
    joined = subject.exact_one_to_one_join(
        left, right, key=subject.SITE_KEY, expected_rows=2
    )
    assert len(joined) == 2
    assert joined["valid_time_utc"].tolist() == [
        1_640_995_200_000_000_000,
        1_640_998_800_000_000_000,
    ]
    duplicate = pd.concat([right, right.iloc[[0]]], ignore_index=True)
    with pytest.raises(subject.IntegrityError, match="duplicate"):
        subject.exact_one_to_one_join(
            left, duplicate, key=subject.SITE_KEY, expected_rows=2
        )
    drift = right.copy()
    drift.loc[1, "site_id"] = "C"
    with pytest.raises(subject.IntegrityError, match="key sets"):
        subject.exact_one_to_one_join(
            left, drift, key=subject.SITE_KEY, expected_rows=2
        )


def test_synthetic_full_join_keeps_group_sources_until_parity_then_releases_tables(
    tmp_path: subject.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = synthetic_sample_schedule()
    timestamps = schedule["valid_time_utc"]
    days = schedule["target_operating_day_kst"]
    timestamp_ordinal = np.arange(subject.EXPECTED_TIMESTAMPS, dtype=np.float64)
    site_ids = np.arange(1, 18, dtype=np.int64)
    site_groups = np.asarray(
        ["kpx_group_1"] * 6
        + ["kpx_group_2"] * 6
        + ["kpx_group_3"] * 5,
        dtype=object,
    )
    group_names = np.asarray(
        ["kpx_group_1", "kpx_group_2", "kpx_group_3"], dtype=object
    )
    site_valid = np.repeat(timestamps, 17)
    group_valid = np.repeat(timestamps, 3)
    site_day = np.repeat(days, 17)
    group_day = np.repeat(days, 3)
    site_base = np.repeat(timestamp_ordinal, 17)
    group_base = np.repeat(timestamp_ordinal, 3)
    site_capacity = np.ones(subject.EXPECTED_SITE_ROWS, dtype=np.float64)
    group_capacity = np.tile(
        np.asarray([6.0, 6.0, 5.0], dtype=np.float64),
        subject.EXPECTED_TIMESTAMPS,
    )
    site_common = {
        "valid_time_utc": site_valid,
        "target_operating_day_kst": site_day,
        "site_id": np.tile(site_ids, subject.EXPECTED_TIMESTAMPS),
        "group": np.tile(site_groups, subject.EXPECTED_TIMESTAMPS),
        "latitude": np.tile(
            np.linspace(33.0, 37.0, 17), subject.EXPECTED_TIMESTAMPS
        ),
        "longitude": np.tile(
            np.linspace(126.0, 130.0, 17), subject.EXPECTED_TIMESTAMPS
        ),
        "capacity_mw": site_capacity,
    }
    group_common = {
        "valid_time_utc": group_valid,
        "target_operating_day_kst": group_day,
        "group": np.tile(group_names, subject.EXPECTED_TIMESTAMPS),
        "site_count": np.tile(
            np.asarray([6, 6, 5], dtype=np.int64),
            subject.EXPECTED_TIMESTAMPS,
        ),
        "capacity_mw": group_capacity,
    }
    external_site = pd.DataFrame(
        {
            **site_common,
            "run_init_utc": np.repeat(schedule["run_init_utc"], 17),
            "forecast_hour": np.repeat(schedule["forecast_hour"], 17),
            **{
                response: site_base + np.float64(index)
                for index, response in enumerate(subject.RESPONSES)
            },
        }
    ).loc[:, subject.EXTERNAL_SITE_COLUMNS]
    external_group = pd.DataFrame(
        {
            **group_common,
            "run_init_utc": np.repeat(schedule["run_init_utc"], 3),
            "forecast_hour": np.repeat(schedule["forecast_hour"], 3),
            **{
                response: group_base + np.float64(index)
                for index, response in enumerate(subject.RESPONSES)
            },
        }
    ).loc[:, subject.EXTERNAL_GROUP_COLUMNS]
    predictor_site = pd.DataFrame(
        {
            **site_common,
            "forecast_kst_dtm": np.repeat(
                schedule["forecast_kst_dtm"], 17
            ),
            "data_available_kst_dtm": np.repeat(
                schedule["data_available_kst_dtm"], 17
            ),
            **{
                predictor: site_base + np.float64(index)
                for index, predictor in enumerate(subject.PREDICTORS)
            },
        }
    ).loc[:, subject.PREDICTOR_SITE_COLUMNS]
    predictor_group = pd.DataFrame(
        {
            **group_common,
            "forecast_kst_dtm": np.repeat(
                schedule["forecast_kst_dtm"], 3
            ),
            "data_available_kst_dtm": np.repeat(
                schedule["data_available_kst_dtm"], 3
            ),
            **{
                predictor: group_base + np.float64(index)
                for index, predictor in enumerate(subject.PREDICTORS)
            },
        }
    ).loc[:, subject.PREDICTOR_GROUP_COLUMNS]
    bound_paths = {
        "decoded_site_matrix": tmp_path / "decoded_site.parquet",
        "decoded_group_matrix": tmp_path / "decoded_group.parquet",
        "bounded_site_predictors": tmp_path / "predictor_site.parquet",
        "bounded_group_predictors": tmp_path / "predictor_group.parquet",
    }
    frame_by_path = {
        str(bound_paths["decoded_site_matrix"]): external_site,
        str(bound_paths["decoded_group_matrix"]): external_group,
        str(bound_paths["bounded_site_predictors"]): predictor_site,
        str(bound_paths["bounded_group_predictors"]): predictor_group,
    }
    table_refs: list[weakref.ReferenceType[object]] = []

    class SyntheticArrowTable:
        def __init__(self, frame: pd.DataFrame) -> None:
            self.frame = frame

        def to_pandas(self) -> pd.DataFrame:
            return self.frame.copy(deep=False)

    def synthetic_read(
        _authorization: object,
        path: subject.Path,
        *_args: object,
        **_kwargs: object,
    ) -> SyntheticArrowTable:
        table = SyntheticArrowTable(frame_by_path[str(path)])
        table_refs.append(weakref.ref(table))
        return table

    monkeypatch.setattr(subject, "guarded_read_table", synthetic_read)
    monkeypatch.setattr(
        subject, "_validate_provenance_alias_schema", lambda *_args: None
    )
    monkeypatch.setattr(
        subject, "_validate_parquet_names_types_rows", lambda *_args, **_kwargs: None
    )
    authorization = subject.PredataAuthorization(
        valid_control(),
        artifact_root=tmp_path,
        alias_files_published=True,
        alias_metadata_validated=True,
    )
    authorization.authorize_arrow_handoff()
    joined = subject.load_and_join_bound_inputs(
        tmp_path,
        bound_paths,
        authorization=authorization,
    )
    assert joined.predictor_matrix_float64.shape == (
        subject.EXPECTED_SITE_ROWS,
        len(subject.PREDICTORS),
    )
    assert [
        int(np.count_nonzero(joined.fold_ordinals == fold.ordinal))
        for fold in subject.FOLDS
    ] == [4_896] * 4
    assert len(joined.group) == subject.EXPECTED_GROUP_ROWS
    gc.collect()
    assert len(table_refs) == 4 and all(ref() is None for ref in table_refs)

    original_group_predictor = predictor_group.loc[0, subject.PREDICTORS[0]]
    predictor_group.loc[0, subject.PREDICTORS[0]] = np.nan
    with pytest.raises(subject.IntegrityError, match="group predictor.*non-finite"):
        subject.load_and_join_bound_inputs(
            tmp_path,
            bound_paths,
            authorization=authorization,
        )
    predictor_group.loc[0, subject.PREDICTORS[0]] = original_group_predictor

    original_available = predictor_site.loc[0, "data_available_kst_dtm"]
    predictor_site.loc[0, "data_available_kst_dtm"] = (
        pd.Timestamp(original_available) + pd.Timedelta(hours=1)
    ).strftime("%Y-%m-%d %H:%M:%S")
    with pytest.raises(subject.IntegrityError, match="availability differs"):
        subject.load_and_join_bound_inputs(
            tmp_path,
            bound_paths,
            authorization=authorization,
        )
    predictor_site.loc[0, "data_available_kst_dtm"] = original_available

    original_group_capacity = predictor_group.loc[0, "capacity_mw"]
    predictor_group.loc[0, "capacity_mw"] = (
        np.float64(original_group_capacity) + np.float64(1e-6)
    )
    with pytest.raises(subject.IntegrityError, match="group metadata parity"):
        subject.load_and_join_bound_inputs(
            tmp_path,
            bound_paths,
            authorization=authorization,
        )
    predictor_group.loc[0, "capacity_mw"] = original_group_capacity
    gc.collect()
    assert len(table_refs) == 16 and all(ref() is None for ref in table_refs)


def test_pooled_quantiles_are_linear_and_same_denominator_drives_cell() -> None:
    actual = np.arange(100, dtype=np.float64)
    pooled = subject.pooled_nrmse_denominator(actual)
    assert pooled.q05 == np.float64(4.95)
    assert pooled.q95 == np.float64(94.05)
    result = subject.regression_metrics(
        actual[:48],
        actual[:48] + np.float64(1.0),
        pooled.denominator,
        minimum_rows=48,
    )
    assert result.valid
    assert result.denominator == pooled.denominator
    assert result.rmse == np.float64(1.0)


@pytest.mark.parametrize(
    ("actual", "reason"),
    [
        (np.zeros(48), "UNUSABLE_CELL_UNIQUE_COUNT_LT_3"),
        (
            np.tile(np.array([0.0, 1e-8, 2e-8]), 16),
            "UNUSABLE_CELL_SST_LTE_1E_MINUS_12",
        ),
    ],
)
def test_metric_degeneracy_is_null_capable_not_nan(
    actual: np.ndarray, reason: str
) -> None:
    result = subject.regression_metrics(
        actual, actual.copy(), np.float64(1.0), minimum_rows=48
    )
    assert not result.valid
    assert result.invalid_reason == reason
    assert result.rmse is None and result.r2 is None and result.nrmse is None


def test_affine_constant_predictor_fallback_and_average_rank_spearman() -> None:
    x = np.ones(12, dtype=np.float64)
    y = np.arange(12, dtype=np.float64)
    fit = subject.analytic_affine_fit_predict(x, y, np.array([1.0, 1.0]))
    assert fit.constant_predictor_branch
    assert fit.slope == np.float64(0.0)
    assert np.array_equal(fit.prediction, np.full(2, np.mean(y)))
    ranked = subject.spearman_correlation(
        np.array([1.0, 1.0, 2.0, 3.0]),
        np.array([10.0, 10.0, 20.0, 30.0]),
    )
    assert ranked.valid and ranked.value == np.float64(1.0)


def test_independent_lower_rmse_and_parent_max_r2_stay_separate() -> None:
    ridge = estimator("RIDGE_PIPELINE", 1.0, 0.2, 0.3)
    trees = estimator("EXTRA_TREES", 1.1, 0.9, 0.2)
    selected = subject.select_estimators(ridge, trees)
    assert selected.independent.estimator == "RIDGE_PIPELINE"
    assert [item.estimator for item in selected.parent_witnesses] == [
        "EXTRA_TREES"
    ]
    with pytest.raises(subject.GlobalAmbiguity, match="ESTIMATOR_TIE"):
        subject.select_estimators(
            estimator("RIDGE_PIPELINE", 1.0, 0.2, 0.3),
            estimator("EXTRA_TREES", 1.0, 0.3, 0.4),
        )


def test_every_applicable_continuous_equality_stops_before_classification() -> None:
    with pytest.raises(subject.GlobalAmbiguity, match="THRESHOLD_EQUALITY"):
        subject.classify_component(
            estimator("RIDGE_PIPELINE", 1.0, 0.995, 0.03),
            estimator("EXTRA_TREES", 2.0, 0.5, 0.5),
            blank_affine_screens(),
            cells_pass=True,
            distribution_pass=True,
        )
    screens = blank_affine_screens()
    screens[0] = subject.AffineScreen(
        subject.PREDICTORS[0], 0.9999, 1.0, 0.001, True
    )
    with pytest.raises(subject.GlobalAmbiguity, match="THRESHOLD_EQUALITY"):
        subject.qualifying_affine_predictors(screens)
    with pytest.raises(subject.GlobalAmbiguity, match="CELL_R2"):
        subject.evaluate_cell_strength(
            metric(0.995, 0.03), weak_floor_applicable=False
        )


def test_affine_screen_is_existential_and_lists_all_hits_in_frozen_order() -> None:
    screens = blank_affine_screens()
    for index in (10, 2):
        screens[index] = subject.AffineScreen(
            subject.PREDICTORS[index], 1.0, -1.0, 0.001, True
        )
    hits = subject.qualifying_affine_predictors(screens)
    assert hits == (subject.PREDICTORS[2], subject.PREDICTORS[10])


def test_strict_component_interior_pass_unstable_and_gray_zone() -> None:
    ridge = estimator("RIDGE_PIPELINE", 1.0, 0.90, 0.11)
    trees = estimator("EXTRA_TREES", 2.0, 0.80, 0.20)
    passed = subject.classify_component(
        ridge,
        trees,
        blank_affine_screens(),
        cells_pass=True,
        distribution_pass=True,
    )
    assert passed.classification == "PASS_NONREDUNDANT_MARGIN"
    unstable = subject.classify_component(
        ridge,
        trees,
        blank_affine_screens(),
        cells_pass=False,
        distribution_pass=True,
    )
    assert unstable.classification == "CLEAR_UNSTABLE_FAILURE"
    with pytest.raises(subject.GlobalAmbiguity, match="GRAY_ZONE"):
        subject.classify_component(
            estimator("RIDGE_PIPELINE", 1.0, 0.994, 0.023),
            estimator("EXTRA_TREES", 2.0, 0.999, 0.20),
            blank_affine_screens(),
            cells_pass=True,
            distribution_pass=True,
        )


def test_season_four_pass_has_no_floor_but_three_pass_remaining_has_floor() -> None:
    years = [metric(0.9, 0.03), metric(0.9, 0.03)]
    forecasts = [metric(0.9, 0.03)] * 3
    groups = OrderedDict(
        (name, metric(0.9, 0.03))
        for name in ("kpx_group_1", "kpx_group_2", "kpx_group_3")
    )
    site_to_group = OrderedDict(
        [
            *[(f"G1_{i}", "kpx_group_1") for i in range(6)],
            *[(f"G2_{i}", "kpx_group_2") for i in range(6)],
            *[(f"G3_{i}", "kpx_group_3") for i in range(5)],
        ]
    )
    sites = OrderedDict((site, metric(0.9, 0.03)) for site in site_to_group)
    four_strict = [metric(0.9, 0.005)] * 4
    gate = subject.evaluate_stability_gate(
        year_metrics=years,
        season_metrics=four_strict,
        forecast_hour_metrics=forecasts,
        group_metrics=groups,
        site_metrics=sites,
        site_to_group=site_to_group,
    )
    assert gate.all_pass
    three_strict = [
        metric(0.9, 0.03),
        metric(0.9, 0.03),
        metric(0.9, 0.03),
        metric(0.999, 0.015),
    ]
    gate = subject.evaluate_stability_gate(
        year_metrics=years,
        season_metrics=three_strict,
        forecast_hour_metrics=forecasts,
        group_metrics=groups,
        site_metrics=sites,
        site_to_group=site_to_group,
    )
    assert gate.season_pass


def test_monthly_iqr_and_psi_duplicate_edge_search_right_semantics() -> None:
    iqr = subject.monthly_iqr_stability(
        np.arange(48, dtype=np.float64),
        np.arange(48, dtype=np.float64) * 2.0,
    )
    assert iqr.valid and iqr.passes and iqr.ratio == np.float64(2.0)
    reference = np.repeat(np.array([0.0, 1.0]), 50)
    comparison = np.repeat(np.array([0.0, 1.0]), 50)
    psi = subject.year_psi(reference, comparison)
    assert psi.valid
    assert psi.final_bin_count == 4
    assert np.array_equal(psi.reference_counts, np.array([0, 50, 0, 50]))
    assert psi.psi == np.float64(0.0)
    with pytest.raises(subject.GlobalAmbiguity, match="IQR"):
        subject.monthly_iqr_stability(
            np.arange(48, dtype=np.float64),
            np.arange(48, dtype=np.float64) * 10.0,
        )


def test_derived_formulas_calm_direction_and_order() -> None:
    raw = {}
    for level in (925, 950, 975, 1000):
        raw[f"UGRD_{level}mb"] = np.array([0.0, float(level)])
        raw[f"VGRD_{level}mb"] = np.array([0.0, 0.0])
    derived = subject.derive_wind_diagnostics(raw)
    assert tuple(derived) == subject.DERIVED_DIAGNOSTICS
    assert derived["ENDPOINT_DU_925_MINUS_1000"][1] == -75.0
    assert derived["ENDPOINT_DV_925_MINUS_1000"][1] == 0.0
    assert derived["ENDPOINT_VECTOR_SHEAR_MAG"][1] == 75.0
    assert derived["ENDPOINT_DIRECTION_COS"][0] == 0.0
    assert derived["ENDPOINT_DIRECTION_SIN"][0] == 0.0
    assert derived["ENDPOINT_DIRECTION_COS"][1] == 1.0


def test_family_max_one_fixed_wind_priority_and_no_component_pruning() -> None:
    results = OrderedDict((name, component("CLEAR_UNSTABLE_FAILURE")) for name in subject.RESPONSES)
    results["HPBL_surface"] = component("PASS_NONREDUNDANT_MARGIN")
    for name in (
        "UGRD_925mb",
        "VGRD_925mb",
        "UGRD_1000mb",
        "VGRD_1000mb",
    ):
        results[name] = component("PASS_NONREDUNDANT_MARGIN")
    endpoints = OrderedDict(
        (name, True) for name in subject.ENDPOINT_VETO_DIAGNOSTICS
    )
    decision = subject.decide_family(results, endpoints)
    assert decision.selected_family == "LOW_LEVEL_ISOBARIC_WIND_PROFILE"
    assert decision.selected_raw_columns == subject.WIND_RESPONSES
    assert decision.selected_derived_columns == subject.DERIVED_DIAGNOSTICS


def test_wind_salvages_constants_but_not_any_physical_failure() -> None:
    results = OrderedDict(
        (name, component("CLEAR_NONNOVEL_CONSTANT"))
        for name in subject.RESPONSES
    )
    results["HPBL_surface"] = component("PASS_NONREDUNDANT_MARGIN")
    for name in (
        "UGRD_925mb",
        "VGRD_925mb",
        "UGRD_1000mb",
        "VGRD_1000mb",
    ):
        results[name] = component("PASS_NONREDUNDANT_MARGIN")
    endpoints = OrderedDict(
        (name, True) for name in subject.ENDPOINT_VETO_DIAGNOSTICS
    )
    assert subject.decide_family(results, endpoints).selected_family == (
        "LOW_LEVEL_ISOBARIC_WIND_PROFILE"
    )
    results["UGRD_950mb"] = component("CLEAR_PHYSICAL_OR_COVERAGE_FAILURE")
    assert subject.decide_family(results, endpoints).selected_family == (
        "PBL_HEIGHT"
    )


class ConstantTree:
    def __init__(self, value: float) -> None:
        self.value = value
        self.check_inputs: list[bool] = []

    def predict(self, x: np.ndarray, *, check_input: bool) -> np.ndarray:
        self.check_inputs.append(check_input)
        assert x.dtype == np.float32 and x.flags.c_contiguous
        return np.full(x.shape[0], self.value, dtype=np.float64)


def test_extra_trees_prediction_is_serial_exact_order_and_repeated() -> None:
    trees = [ConstantTree(float(index)) for index in range(128)]
    forest = type("Forest", (), {"estimators_": trees})()
    x = np.zeros((4, 35), dtype=np.float64)
    prediction = subject.serial_extra_trees_predict(forest, x)
    assert np.array_equal(prediction, np.full(4, 63.5))
    assert all(tree.check_inputs == [False, False] for tree in trees)


def test_manual_ridge_prediction_matches_preconstructed_attributes_no_fit() -> None:
    rng = np.random.default_rng(260810)
    x = rng.normal(size=(96, 35)).astype(np.float64)
    mean = np.linspace(-0.5, 0.5, 35, dtype=np.float64)
    scale = np.linspace(0.75, 1.75, 35, dtype=np.float64)
    coefficient = np.linspace(-1.0, 1.0, 35, dtype=np.float64)
    intercept = np.float64(0.5)
    scaler = type("PreconstructedScaler", (), {"mean_": mean, "scale_": scale})()
    ridge = type(
        "PreconstructedRidge",
        (),
        {"coef_": coefficient, "intercept_": intercept},
    )()
    pipeline = type(
        "PreconstructedPipeline",
        (),
        {
            "named_steps": OrderedDict(
                (("standard_scaler", scaler), ("ridge", ridge))
            )
        },
    )()
    expected = np.sum(
        ((x[:8] - mean) / scale) * coefficient,
        axis=1,
        dtype=np.float64,
    ) + intercept
    observed = subject.manual_ridge_pipeline_predict(pipeline, x[:8])
    assert np.array_equal(observed, expected)


def test_preseal_v5_test_source_has_no_executable_fit_calls() -> None:
    tree = ast.parse(subject.Path(__file__).read_text(encoding="utf-8"))
    fit_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit"
    ]
    assert fit_calls == []


def test_float_hex_is_binary64_and_signed_zero_persists_positive() -> None:
    assert subject.canonical_float_hex(np.float64(-0.0)) == "0x0.0p+0"
    value, witness = subject.persist_float(np.float64(0.1))
    assert value == 0.1
    assert witness == float(np.float64(0.1)).hex().lower()


def test_decision_validator_derives_cutoff_hex_equality_and_reason_order() -> None:
    rows = [raw_result(response) for response in subject.RESPONSES]
    rows[0] = raw_result("HPBL_surface", "AMBIGUOUS_THRESHOLD_EQUALITY")
    document = decision_document(rows)
    witness = subject.make_threshold_witness(
        cutoff_name="INDEPENDENT_R2_STABLE_MARGIN_0.9925",
        scope="RAW/HPBL_surface/INDEPENDENT/RIDGE_PIPELINE",
        diagnostic="HPBL_surface",
        value=np.float64(0.9925),
        cutoff=0.9925,
    )
    document["all_threshold_float_hex_witnesses"] = [witness]
    document["global_ambiguity"] = True
    document["global_ambiguity_reasons"] = [
        "AMBIGUOUS_THRESHOLD_EQUALITY:"
        + witness["cutoff_name"]
        + ":"
        + witness["scope"]
        + ":"
        + witness["diagnostic"]
        + ":"
        + witness["value_float_hex"]
    ]
    document["status"] = "STOP_FAMILY_AMBIGUOUS"
    document["selected_family"] = None
    document["selected_raw_columns"] = []
    document["selected_derived_columns"] = []
    subject.validate_family_decision_document(document)
    broken = dict(document)
    broken_witness = dict(witness)
    broken_witness["cutoff_float_hex"] = "0x0.0p+0"
    broken["all_threshold_float_hex_witnesses"] = [broken_witness]
    with pytest.raises(subject.IntegrityError, match="cutoff float hex"):
        subject.validate_family_decision_document(broken)
    broken = dict(document)
    broken_witness = dict(witness)
    broken_witness["boundary_equal"] = False
    broken["all_threshold_float_hex_witnesses"] = [broken_witness]
    with pytest.raises(subject.IntegrityError, match="direct equality"):
        subject.validate_family_decision_document(broken)


def test_decision_validator_crosschecks_endpoint_binding_and_family_derivation() -> None:
    document = decision_document()
    subject.validate_family_decision_document(document)
    broken = dict(document)
    endpoints = [dict(item) for item in document["endpoint_veto_results"]]
    endpoints[0]["source_component_estimator_binding"] = {
        "UGRD_925mb": "RIDGE_PIPELINE",
        "UGRD_1000mb": None,
    }
    broken["endpoint_veto_results"] = endpoints
    with pytest.raises(subject.IntegrityError, match="raw independent selector"):
        subject.validate_family_decision_document(broken)
    broken = dict(document)
    broken["family_clear_pass_results"] = {
        "LOW_LEVEL_ISOBARIC_WIND_PROFILE": False,
        "PBL_HEIGHT": True,
    }
    with pytest.raises(subject.IntegrityError, match="family clear"):
        subject.validate_family_decision_document(broken)


def test_decision_validator_enforces_wind_priority_when_both_clear_pass() -> None:
    rows = [
        raw_result(response, "PASS_NONREDUNDANT_MARGIN")
        for response in subject.RESPONSES
    ]
    document = decision_document(rows, endpoints_pass=True)
    assert document["selected_family"] == "LOW_LEVEL_ISOBARIC_WIND_PROFILE"
    subject.validate_family_decision_document(document)
    document["selected_family"] = "PBL_HEIGHT"
    document["selected_raw_columns"] = list(subject.PBL_RESPONSES)
    document["selected_derived_columns"] = []
    with pytest.raises(subject.IntegrityError, match="PBL selected"):
        subject.validate_family_decision_document(document)


def test_parent_tie_is_masked_only_by_prior_affine_redundancy() -> None:
    rows = [raw_result(response) for response in subject.RESPONSES]
    masked = raw_result("HPBL_surface", "CLEAR_REDUNDANT_AFFINE")
    masked["parent_reference_estimators"] = [
        "RIDGE_PIPELINE",
        "EXTRA_TREES",
    ]
    masked["parent_redundancy_boolean"] = None
    masked["affine_qualifying_predictors"] = [subject.PREDICTORS[0]]
    rows[0] = masked
    document = decision_document(rows)
    tie = {
        "tie_kind": "PARENT_MAX_R2",
        "diagnostic": "HPBL_surface",
        "estimators": ["RIDGE_PIPELINE", "EXTRA_TREES"],
        "metric_name": "R2",
        "metric_values_float_hex": [
            float(0.9).hex().lower(),
            float(0.9).hex().lower(),
        ],
        "resolution": "MASKED_BY_PRIOR_AFFINE_REDUNDANCY",
        "global_ambiguity": False,
    }
    document["all_tied_witnesses"] = [tie]
    subject.validate_family_decision_document(document)
    tie["resolution"] = "GLOBAL_AMBIGUITY"
    tie["global_ambiguity"] = True
    with pytest.raises(subject.IntegrityError, match="global parent tie"):
        subject.validate_family_decision_document(document)


@pytest.mark.parametrize(
    "attempt_id",
    [
        "target_free_duplicate_v5__20260811T123456123456Z",
        "target_free_duplicate_v5__20260229T123456123456Z",
        "target_free_duplicate_v5__20260811T12345612345Z",
    ],
)
def test_attempt_id_regex_and_calendar_validation(attempt_id: str) -> None:
    if attempt_id == "target_free_duplicate_v5__20260811T123456123456Z":
        assert subject.validate_attempt_id(attempt_id) == attempt_id
    else:
        with pytest.raises(subject.PredataAuthorizationError):
            subject.validate_attempt_id(attempt_id)


def test_materializers_freeze_decision_ledger_and_distribution_counts() -> None:
    raw = constant_raw_evaluations()
    derived = empty_derived_evaluation(raw)
    decision = subject.build_family_decision_document(raw, derived)
    assert decision["status"] == "STOP_NO_TARGET_FREE_INCREMENTAL_FAMILY"
    assert decision["selected_family"] is None
    ledger = subject.build_fit_ledger_document(
        raw, decision_status=str(decision["status"])
    )
    assert tuple(ledger) == subject.FIT_LEDGER_TOP_KEYS
    assert ledger["executed_counts"] == {
        "completed_primary_model_units": 0,
        "completed_analytic_affine_units": 0,
        "completed_total_decision_units": 0,
    }
    assert ledger["skipped_counts"] == {
        "skipped_primary_model_units": 72,
        "skipped_analytic_affine_units": 1260,
        "skipped_total_decision_units": 1332,
    }
    distribution = subject.build_distribution_document(
        raw, derived, decision_status=str(decision["status"])
    )
    assert tuple(distribution) == subject.DISTRIBUTION_TOP_KEYS
    assert len(distribution["records"]) == 624
    assert len(distribution["endpoint_pooled_metric_records"]) == 3


def test_fit_ledger_validator_rejects_partial_response_and_slot_tampering() -> None:
    raw = completed_raw_evaluations()
    ledger = subject.build_fit_ledger_document(
        raw,
        decision_status=(
            "PASS_ONE_TARGET_FREE_FAMILY_SELECTED_PENDING_INDEPENDENT_POSTRUN_SEAL"
        ),
    )

    partial = copy.deepcopy(ledger)
    first_unit = subject.planned_fit_units()[0]
    partial["unit_slots"][0] = subject._skipped_primary_record(
        first_unit,
        "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3",
    )
    partial["executed_counts"] = {
        "completed_primary_model_units": 71,
        "completed_analytic_affine_units": 1260,
        "completed_total_decision_units": 1331,
    }
    partial["skipped_counts"] = {
        "skipped_primary_model_units": 1,
        "skipped_analytic_affine_units": 0,
        "skipped_total_decision_units": 1,
    }
    partial["internal_method_call_counts"][
        "sklearn.pipeline.Pipeline.fit"
    ] -= 1
    partial["internal_method_call_counts"][
        "sklearn.preprocessing.StandardScaler.fit"
    ] -= 1
    partial["internal_method_call_counts"][
        "sklearn.linear_model.Ridge.fit"
    ] -= 1
    partial["internal_method_call_counts"]["total_method_calls"] -= 3
    with pytest.raises(subject.IntegrityError, match="partially completed"):
        subject.validate_fit_ledger_document(partial)

    bad_method_total = copy.deepcopy(ledger)
    bad_method_total["internal_method_call_counts"][
        "sklearn.pipeline.Pipeline.fit"
    ] += 1
    bad_method_total["internal_method_call_counts"]["total_method_calls"] += 1
    with pytest.raises(subject.IntegrityError, match="method counts differ"):
        subject.validate_fit_ledger_document(bad_method_total)

    bad_affine_hex = copy.deepcopy(ledger)
    bad_affine_hex["unit_slots"][subject.EXPECTED_PRIMARY_UNITS][
        "sxx_float_hex"
    ] = float(2.0).hex().lower()
    with pytest.raises(subject.IntegrityError, match="numeric witness differs"):
        subject.validate_fit_ledger_document(bad_affine_hex)

    bad_constant_branch = copy.deepcopy(ledger)
    affine = bad_constant_branch["unit_slots"][subject.EXPECTED_PRIMARY_UNITS]
    affine["sxx"] = 0.0
    affine["sxx_float_hex"] = float(0.0).hex().lower()
    affine["constant_predictor_branch"] = False
    with pytest.raises(subject.IntegrityError, match="constant-predictor branch"):
        subject.validate_fit_ledger_document(bad_constant_branch)


def test_published_alias_closure_is_atomic_before_arrow_handoff(
    tmp_path: subject.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization, identities = alias_closure_fixture(tmp_path, monkeypatch)
    assert authorization.validation.aliases_pass is False
    assert authorization.alias_metadata_validated is False
    assert authorization.arrow_handoff_started is False
    subject.validate_published_alias_closure(
        tmp_path, identities, authorization=authorization
    )
    assert authorization.validation.aliases_pass is True
    assert authorization.alias_metadata_validated is True
    assert authorization.arrow_handoff_started is False
    authorization.authorize_arrow_handoff()
    assert authorization.arrow_handoff_started is True


@pytest.mark.parametrize(
    ("semantic_tamper", "provenance_rows", "missing_field", "match"),
    [
        (True, 10_368, False, "semantic closure"),
        (False, 10_367, False, "row count"),
        (False, 10_368, True, "lacks exact15"),
    ],
)
def test_published_alias_closure_failure_never_opens_value_handoff(
    tmp_path: subject.Path,
    monkeypatch: pytest.MonkeyPatch,
    semantic_tamper: bool,
    provenance_rows: int,
    missing_field: bool,
    match: str,
) -> None:
    authorization, identities = alias_closure_fixture(
        tmp_path,
        monkeypatch,
        semantic_tamper=semantic_tamper,
        provenance_rows=provenance_rows,
        missing_provenance_field=missing_field,
    )
    with pytest.raises(subject.ContractError, match=match):
        subject.validate_published_alias_closure(
            tmp_path, identities, authorization=authorization
        )
    assert authorization.validation.aliases_pass is False
    assert authorization.alias_metadata_validated is False
    assert authorization.arrow_handoff_started is False
    with pytest.raises(subject.PredataAuthorizationError):
        authorization.authorize_arrow_handoff()


def test_endpoint_raw_tie_uses_exact13_reason_in_every_evidence_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_reasons = (
        "SKIPPED_PREMODEL_CLEAR_PHYSICAL_WIND_FAMILY_VETO",
        "SKIPPED_PREMODEL_CLEAR_PHYSICAL_PBL_FAMILY_VETO",
        "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3",
        "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_POOLED_SST_LTE_1E_MINUS_12",
        "UNUSABLE_CELL_UNIQUE_COUNT_LT_3",
        "UNUSABLE_CELL_SST_LTE_1E_MINUS_12",
        "UNUSABLE_MONTHLY_IQR_2022_LTE_1E_MINUS_12",
        "UNUSABLE_MONTHLY_IQR_2023_LTE_1E_MINUS_12",
        "UNUSABLE_MONTHLY_IQR_BOTH_YEARS_LTE_1E_MINUS_12",
        "UNUSABLE_PSI_FINAL_BIN_COUNT_LT_2",
        "DEGENERATE_AFFINE_PREDICTOR_SXX_LTE_1E_MINUS_12",
        "DEGENERATE_SPEARMAN_RANK_VARIANCE_LTE_1E_MINUS_12",
        "STRUCTURALLY_NOT_APPLICABLE",
    )
    assert subject.INVALID_OR_SKIP_REASONS == expected_reasons
    raw = constant_raw_evaluations()
    for response in subject.WIND_RESPONSES:
        raw[response] = subject.dataclasses.replace(
            raw[response],
            raw_result=raw_result(
                response, "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
            ),
        )

    def fake_derived(_components: object) -> dict[str, np.ndarray]:
        return {
            diagnostic: np.zeros(
                subject.EXPECTED_SITE_ROWS, dtype=np.float64
            )
            for diagnostic in subject.DERIVED_DIAGNOSTICS
        }

    monkeypatch.setattr(subject, "derive_partial_wind_diagnostics", fake_derived)
    monkeypatch.setattr(
        subject,
        "_capacity_aggregate_component_mapping",
        lambda _site, components: (object(), dict(components)),
    )
    joined = type("SyntheticJoined", (), {"site": object()})()
    prepared = type("SyntheticPrepared", (), {"joined": joined})()
    derived = subject.evaluate_derived_wind(prepared, raw)
    assert [row["verdict"] for row in derived.endpoint_results] == [
        "AMBIGUOUS_RAW_ESTIMATOR_TIE"
    ] * 3
    for record in (
        *derived.endpoint_pooled_records,
        *derived.stability_records,
        *derived.distribution_records,
    ):
        assert record["valid"] is False
        assert record["invalid_reason"] == "STRUCTURALLY_NOT_APPLICABLE"
        subject.validate_validity_reason(
            record["valid"], record["invalid_reason"], "endpoint evidence"
        )
    with pytest.raises(subject.IntegrityError, match="outside exact13"):
        subject.validate_validity_reason(
            False,
            "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE",
            "endpoint evidence",
        )


def test_execution_budget_rejects_every_legacy_key_name() -> None:
    assert subject.PROCESS_CAPTURE_KEYS == (
        "command",
        "exit_code",
        "stdout_payload",
        "stdout_size_bytes",
        "stdout_sha256",
        "stderr_size_bytes",
        "stderr_sha256",
        "wall_seconds_observed",
        "full_stdout_identity_instrumented",
    )
    assert subject.POSTRUN_AUDIT_RESULT_KEYS == (
        *subject.PROCESS_CAPTURE_KEYS,
        "runner_result",
    )
    subject.validate_execution_budget(dict(subject.EXECUTION_BUDGET))
    legacy = {
        "attempt_number": 1,
        "alias_files": 2,
        "primary_model_fit_units": 72,
        "analytic_affine_fit_units": 1260,
        "total_fit_ledger_units": 1332,
        "fit_ledger_rows": 1332,
        "oof_rows": 352512,
        "metrics_rows": 333,
        "stability_rows": 348,
        "distribution_records": 624,
        "endpoint_pooled_metric_records": 3,
        "derived_rows": 414720,
        "runner_staged_outputs": 7,
        "final_complete_files": 11,
    }
    with pytest.raises(subject.PredataAuthorizationError, match="key set"):
        subject.validate_execution_budget(legacy)


def test_runtime_caps_are_checked_before_every_numeric_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = inspect.getsource(subject.validate_runtime_identity)
    guard_offset = source.index("assert_runtime_environment_caps(os.environ)")
    for import_literal in (
        "import joblib",
        "import numpy",
        "import pandas",
        "import pyarrow",
        "import scipy",
        "import sklearn",
        "import threadpoolctl",
    ):
        assert guard_offset < source.index(import_literal)
    for key, value in subject.ENVIRONMENT_CAPS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    numeric_imports: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".", 1)[0] in {
            "joblib",
            "numpy",
            "pandas",
            "pyarrow",
            "scipy",
            "sklearn",
            "threadpoolctl",
        }:
            numeric_imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(subject.PredataAuthorizationError, match="OMP_NUM_THREADS"):
        subject.validate_runtime_identity()
    assert numeric_imports == []


def test_threadpool_exact_metadata_and_count_tamper_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threadpoolctl

    records = []
    for name, identity in subject.NATIVE_LIBRARY_IDENTITIES.items():
        metadata = dict(subject.THREADPOOL_METADATA_EXACT[name])
        records.append(
            {
                "filepath": identity.path,
                "num_threads": 1,
                **metadata,
            }
        )
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: records)
    observed = subject.capture_threadpool_info_exact(expected_threads=1)
    assert [item["name"] for item in observed] == list(
        subject.NATIVE_LIBRARY_IDENTITIES
    )
    records[0] = {**records[0], "num_threads": 2}
    with pytest.raises(subject.PredataAuthorizationError, match="count differs"):
        subject.capture_threadpool_info_exact(expected_threads=1)


def test_threadpool_recorder_requires_balanced_inside_after_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = [
        {
            "name": name,
            "path": subject.NATIVE_LIBRARY_IDENTITIES[name].path,
            **dict(subject.THREADPOOL_METADATA_EXACT[name]),
            "num_threads": 1,
        }
        for name in subject.NATIVE_LIBRARY_IDENTITIES
    ]
    monkeypatch.setattr(
        subject,
        "capture_threadpool_info_exact",
        lambda *, expected_threads: observation,
    )
    recorder = subject.ThreadpoolCaptureRecorder()
    recorder.capture(
        phase="BEFORE_EXPLICIT_CONTEXT", label="before", observed=observation
    )
    recorder.capture(phase="INSIDE_CONTEXT", label="fit")
    with pytest.raises(subject.PredataAuthorizationError, match="inside/after"):
        recorder.snapshot()
    recorder.capture(phase="AFTER_CONTEXT", label="fit")
    snapshot = recorder.snapshot()
    assert snapshot["capture_counts"] == {
        "BEFORE_EXPLICIT_CONTEXT": 1,
        "INSIDE_CONTEXT": 1,
        "AFTER_CONTEXT": 1,
    }
    assert len(snapshot["event_chain_sha256"]) == 64
    zero = subject.ThreadpoolCaptureRecorder()
    zero.capture(
        phase="BEFORE_EXPLICIT_CONTEXT", label="before", observed=observation
    )
    raw = constant_raw_evaluations()
    ledger = subject.build_fit_ledger_document(
        raw, decision_status="STOP_NO_TARGET_FREE_INCREMENTAL_FAMILY"
    )
    subject.validate_threadpool_evidence(zero.snapshot(), ledger)


def test_full_fit_threadpool_count_is_ledger_derived_and_not_substitutable() -> None:
    raw = completed_raw_evaluations()
    ledger = subject.build_fit_ledger_document(
        raw,
        decision_status=(
            "PASS_ONE_TARGET_FREE_FAMILY_SELECTED_PENDING_INDEPENDENT_POSTRUN_SEAL"
        ),
    )
    evidence = zero_threadpool_evidence()
    observation = evidence["first_observation_by_phase"][
        "BEFORE_EXPLICIT_CONTEXT"
    ]["threadpools"]
    expected_inside = 2 * 72 + 1260

    def event(ordinal: int, phase: str, label: str) -> dict[str, object]:
        return {
            "event_ordinal": ordinal,
            "phase": phase,
            "label": label,
            "threadpools": observation,
        }

    evidence["capture_counts"] = {
        "BEFORE_EXPLICIT_CONTEXT": 1,
        "INSIDE_CONTEXT": expected_inside,
        "AFTER_CONTEXT": expected_inside,
    }
    evidence["event_count"] = 1 + 2 * expected_inside
    evidence["first_observation_by_phase"]["INSIDE_CONTEXT"] = event(
        2, "INSIDE_CONTEXT", "first-inside"
    )
    evidence["first_observation_by_phase"]["AFTER_CONTEXT"] = event(
        3, "AFTER_CONTEXT", "first-after"
    )
    evidence["last_observation_by_phase"]["INSIDE_CONTEXT"] = event(
        2 * expected_inside, "INSIDE_CONTEXT", "last-inside"
    )
    evidence["last_observation_by_phase"]["AFTER_CONTEXT"] = event(
        1 + 2 * expected_inside, "AFTER_CONTEXT", "last-after"
    )
    subject.validate_threadpool_evidence(evidence, ledger)
    evidence["capture_counts"]["INSIDE_CONTEXT"] -= 1
    with pytest.raises(subject.IntegrityError, match="count crosslink"):
        subject.validate_threadpool_evidence(evidence, ledger)


def test_control_timestamps_require_exact_microseconds_and_strict_order() -> None:
    assert subject._rfc3339_100ns_ticks(
        "2026-08-11T12:00:00.000001Z", "valid"
    ) > 0
    for invalid in (
        "2026-08-11T12:00:00.00001Z",
        "2026-08-11T12:00:00.0000001Z",
    ):
        with pytest.raises(
            subject.PredataAuthorizationError, match="exactly 6"
        ):
            subject._rfc3339_100ns_ticks(invalid, "invalid")
    amendment = {"created_utc": "2026-08-11T12:00:00.000001Z"}
    amendment_v4 = {"created_utc": "2026-08-11T12:00:00.000002Z"}
    historical = {
        role: {"created_utc": f"2026-08-11T12:00:00.00000{index}Z"}
        for role, index in zip(
            ("code_seal", "authorization", "independent_review", "independent_go"),
            (3, 4, 5, 6),
            strict=True,
        )
    }
    incident = {"created_utc": "2026-08-11T12:00:00.000007Z"}
    amendment_v5 = {"created_utc": "2026-08-11T12:00:00.000008Z"}
    controls = {
        "CODE_SEAL": {
            "created_utc": "2026-08-11T12:00:00.000010Z",
            "test_evidence": {
                "completed_utc": "2026-08-11T12:00:00.000009Z"
            },
        },
        "AUTHORIZATION": {"created_utc": "2026-08-11T12:00:00.000011Z"},
        "REVIEW": {"created_utc": "2026-08-11T12:00:00.000012Z"},
        "GO": {"created_utc": "2026-08-11T12:00:00.000013Z"},
    }
    args = (amendment, amendment_v4, amendment_v5, incident, historical)
    subject.validate_control_chronology(*args, controls)
    equal_completion_and_seal = copy.deepcopy(controls)
    equal_completion_and_seal["CODE_SEAL"]["created_utc"] = controls[
        "CODE_SEAL"
    ]["test_evidence"]["completed_utc"]
    subject.validate_control_chronology(
        *args, equal_completion_and_seal
    )
    for seal_time in (
        amendment_v5["created_utc"],
        incident["created_utc"],
    ):
        broken = copy.deepcopy(controls)
        broken["CODE_SEAL"]["created_utc"] = seal_time
        with pytest.raises(subject.PredataAuthorizationError, match="not strict"):
            subject.validate_control_chronology(
                *args, broken
            )
    broken_v5 = {"created_utc": incident["created_utc"]}
    with pytest.raises(subject.PredataAuthorizationError, match="not strict"):
        subject.validate_control_chronology(
            amendment, amendment_v4, broken_v5, incident, historical, controls
        )


def test_code_seal_test_evidence_rejects_every_common_drift(
    tmp_path: subject.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = {role: [role, "synthetic"] for role in subject.CONTROL_COMMAND_KEYS}
    monkeypatch.setattr(subject, "expected_test_commands", lambda _root: commands)
    results = {
        role: {
            "test_file": subject.TEST_FILES[role],
            "exit_code": 0,
            "passed": 3,
            "skipped": 0,
            "deselected": 0,
            "summary": "3 passed in 1.25s",
        }
        for role in subject.CONTROL_COMMAND_KEYS
    }
    record = {
        "created_utc": "2026-08-11T12:00:00.000002Z",
        "test_evidence": {
            "commands": copy.deepcopy(commands),
            "results": results,
            "all_exit_codes_zero": True,
            "all_expected_test_files_covered": True,
            "network_denied": True,
            "pycache_disabled": True,
            "code_identities_revalidated": True,
            "completed_utc": "2026-08-11T12:00:00.000001Z",
        },
        "required_next_controls": dict(subject.REQUIRED_NEXT_CONTROLS),
    }
    subject._validate_code_seal_nested(record, tmp_path)
    mutations = []
    wrong_command = copy.deepcopy(record)
    wrong_command["test_evidence"]["commands"]["runner"] = ["arbitrary"]
    mutations.append((wrong_command, "command argv"))
    wrong_file = copy.deepcopy(record)
    wrong_file["test_evidence"]["results"]["runner"]["test_file"] = "wrong.py"
    mutations.append((wrong_file, "file differs"))
    negative = copy.deepcopy(record)
    negative["test_evidence"]["results"]["runner"]["skipped"] = -1
    mutations.append((negative, "count type"))
    zero_pass = copy.deepcopy(record)
    zero_pass["test_evidence"]["results"]["runner"]["passed"] = 0
    zero_pass["test_evidence"]["results"]["runner"]["summary"] = (
        "0 passed in 1.25s"
    )
    mutations.append((zero_pass, "must be positive"))
    malformed = copy.deepcopy(record)
    malformed["test_evidence"]["results"]["runner"]["summary"] = "PASS"
    mutations.append((malformed, "summary differs"))
    mismatched_summary = copy.deepcopy(record)
    mismatched_summary["test_evidence"]["results"]["runner"]["summary"] = (
        "2 passed in 1.25s"
    )
    mutations.append((mismatched_summary, "count crosslink"))
    suffix_absent = copy.deepcopy(record)
    suffix_absent["test_evidence"]["results"]["runner"]["summary"] = (
        "3 passed in 61.25s"
    )
    subject._validate_code_seal_nested(suffix_absent, tmp_path)
    rounded_suffix = copy.deepcopy(record)
    rounded_suffix["test_evidence"]["results"]["runner"]["summary"] = (
        "3 passed in 61.00s (0:01:00)"
    )
    subject._validate_code_seal_nested(rounded_suffix, tmp_path)
    wrong_long_clock = copy.deepcopy(record)
    wrong_long_clock["test_evidence"]["results"]["runner"]["summary"] = (
        "3 passed in 61.25s (0:01:02)"
    )
    mutations.append((wrong_long_clock, "clock suffix differs"))
    fractional_rounding_mismatch = copy.deepcopy(record)
    fractional_rounding_mismatch["test_evidence"]["results"]["runner"][
        "summary"
    ] = "3 passed in 61.01s (0:01:00)"
    mutations.append((fractional_rounding_mismatch, "clock suffix differs"))
    subminute_suffix = copy.deepcopy(record)
    subminute_suffix["test_evidence"]["results"]["runner"]["summary"] = (
        "3 passed in 60.00s (0:00:59)"
    )
    mutations.append((subminute_suffix, "clock suffix differs"))
    noncanonical_hours = copy.deepcopy(record)
    noncanonical_hours["test_evidence"]["results"]["runner"]["summary"] = (
        "3 passed in 3600.00s (01:00:00)"
    )
    mutations.append((noncanonical_hours, "clock suffix differs"))
    after_seal = copy.deepcopy(record)
    after_seal["test_evidence"]["completed_utc"] = (
        "2026-08-11T12:00:00.000003Z"
    )
    mutations.append((after_seal, "after code seal"))
    for broken, match in mutations:
        with pytest.raises(subject.PredataAuthorizationError, match=match):
            subject._validate_code_seal_nested(broken, tmp_path)


def test_pre_run_zero_state_rejects_preexisting_postrun_and_output(
    tmp_path: subject.Path,
) -> None:
    postrun = tmp_path / subject.POSTRUN_PASS_RELATIVE_PATH
    postrun.parent.mkdir(parents=True)
    postrun.write_bytes(b"stale")
    with pytest.raises(subject.PredataAuthorizationError, match="postrun PASS"):
        subject.validate_runner_zero_state(tmp_path)
    postrun.unlink()
    (tmp_path / subject.OUTPUT_ROOT_RELATIVE).mkdir(parents=True)
    with pytest.raises(subject.PredataAuthorizationError, match="output namespace"):
        subject.validate_runner_zero_state(tmp_path)
    (tmp_path / subject.OUTPUT_ROOT_RELATIVE).rmdir()
    future_incident = tmp_path / subject.V5_FAILURE_INCIDENT_RELATIVE_PATH
    future_incident.parent.mkdir(parents=True, exist_ok=True)
    future_incident.write_bytes(b"stale")
    with pytest.raises(subject.PredataAuthorizationError, match="future V5"):
        subject.validate_runner_zero_state(tmp_path)


def _make_test_symlink(
    link: subject.Path, target: subject.Path, *, target_is_directory: bool = False
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    assert subject._lexists(link)


def _synthetic_cli_control_paths(
    root: subject.Path,
) -> tuple[subject.Path, subject.Path]:
    authorization = root / subject.EXECUTION_AUTHORIZATION_RELATIVE_PATH
    independent_go = root / subject.INDEPENDENT_GO_RELATIVE_PATH
    authorization.parent.mkdir(parents=True, exist_ok=True)
    independent_go.parent.mkdir(parents=True, exist_ok=True)
    authorization.write_bytes(b"authorization")
    independent_go.write_bytes(b"independent-go")
    return authorization, independent_go


def test_production_cli_paths_require_exact_absolute_lexemes(
    tmp_path: subject.Path,
) -> None:
    root = tmp_path / "artifact-root"
    root.mkdir()
    authorization, independent_go = _synthetic_cli_control_paths(root)
    assert subject.validate_production_cli_paths(
        str(root),
        str(authorization),
        str(independent_go),
        expected_root=root,
    ) == (root, authorization, independent_go)

    relative_cases = (
        (root.name, str(authorization), str(independent_go)),
        (str(root), subject.EXECUTION_AUTHORIZATION_RELATIVE_PATH, str(independent_go)),
        (str(root), str(authorization), subject.INDEPENDENT_GO_RELATIVE_PATH),
    )
    for supplied_root, supplied_authorization, supplied_go in relative_cases:
        with pytest.raises(
            subject.PredataAuthorizationError, match="must be absolute"
        ):
            subject.validate_production_cli_paths(
                supplied_root,
                supplied_authorization,
                supplied_go,
                expected_root=root,
            )

    alias_root = root.parent / root.name / ".." / root.name
    alias_authorization = authorization.parent / ".." / authorization.parent.name / authorization.name
    alias_go = independent_go.parent / ".." / independent_go.parent.name / independent_go.name
    for supplied_root, supplied_authorization, supplied_go in (
        (str(alias_root), str(authorization), str(independent_go)),
        (str(root), str(alias_authorization), str(independent_go)),
        (str(root), str(authorization), str(alias_go)),
    ):
        with pytest.raises(
            subject.PredataAuthorizationError, match="canonical lexical path"
        ):
            subject.validate_production_cli_paths(
                supplied_root,
                supplied_authorization,
                supplied_go,
                expected_root=root,
            )


def test_production_cli_paths_reject_root_authorization_and_go_links(
    tmp_path: subject.Path,
) -> None:
    target_root = tmp_path / "target-root"
    target_root.mkdir()
    target_authorization, target_go = _synthetic_cli_control_paths(target_root)

    linked_root = tmp_path / "linked-root"
    _make_test_symlink(linked_root, target_root, target_is_directory=True)
    with pytest.raises(subject.PredataAuthorizationError, match="link-like"):
        subject.validate_production_cli_paths(
            str(linked_root),
            str(linked_root / subject.EXECUTION_AUTHORIZATION_RELATIVE_PATH),
            str(linked_root / subject.INDEPENDENT_GO_RELATIVE_PATH),
            expected_root=linked_root,
        )
    linked_root.unlink()

    target_authorization.unlink()
    authorization_target = tmp_path / "authorization-target.json"
    authorization_target.write_bytes(b"authorization")
    _make_test_symlink(target_authorization, authorization_target)
    with pytest.raises(subject.PredataAuthorizationError, match="link-like"):
        subject.validate_production_cli_paths(
            str(target_root),
            str(target_authorization),
            str(target_go),
            expected_root=target_root,
        )
    target_authorization.unlink()
    target_authorization.write_bytes(b"authorization")

    target_go.unlink()
    go_target = tmp_path / "go-target.json"
    go_target.write_bytes(b"go")
    _make_test_symlink(target_go, go_target)
    with pytest.raises(subject.PredataAuthorizationError, match="link-like"):
        subject.validate_production_cli_paths(
            str(target_root),
            str(target_authorization),
            str(target_go),
            expected_root=target_root,
        )


def test_linklike_detects_generic_windows_reparse_attribute() -> None:
    class ReparseMetadata:
        st_file_attributes = 0x400

    class ReparsePath:
        @staticmethod
        def is_symlink() -> bool:
            return False

        @staticmethod
        def is_junction() -> bool:
            return False

        @staticmethod
        def lstat() -> ReparseMetadata:
            return ReparseMetadata()

    assert subject._linklike(ReparsePath()) is True


def test_actual_runner_command_is_type_and_position_exact() -> None:
    expected = [
        r"C:\repo\.venv\Scripts\python.exe",
        "-B",
        "-m",
        "scripts.run_noaa_gfs_target_free_duplicate_v5",
        "--root",
        r"C:\repo\artifacts\frozen",
        "--authorization",
        r"C:\repo\artifacts\frozen\prereg\authorization.json",
        "--independent-go",
        r"C:\repo\artifacts\frozen\independent_redteam\go.json",
    ]
    assert subject.validate_actual_runner_command(expected, expected) == expected

    mutations: list[object] = [
        None,
        [*expected, "--extra"],
        [*expected[:4], "--authorization", expected[7], "--root", expected[5], *expected[8:]],
        [".venv/Scripts/python.exe", *expected[1:]],
        tuple(expected),
        [*expected[:1], False, *expected[2:]],
    ]
    for actual in mutations:
        with pytest.raises(subject.PredataAuthorizationError):
            subject.validate_actual_runner_command(actual, expected)

    relative_authorized = [".venv/Scripts/python.exe", *expected[1:]]
    with pytest.raises(
        subject.PredataAuthorizationError, match="interpreter path must be absolute"
    ):
        subject.validate_actual_runner_command(expected, relative_authorized)


def test_actual_runner_command_guard_is_before_runtime_alias_and_value_handoff() -> None:
    chain_source = inspect.getsource(subject.validate_execution_control_chain)
    assert chain_source.index("_load_control(") < chain_source.index(
        "validate_actual_runner_command("
    )
    assert chain_source.index("required command argv differs") < chain_source.index(
        "validate_actual_runner_command("
    )
    preparation_source = inspect.getsource(subject.prepare_execution)
    chain_index = preparation_source.index("validate_execution_control_chain(")
    assert chain_index < preparation_source.index("validate_runtime_identity()")
    gate_index = preparation_source.index("validate_pre_root_source_metadata_gate(")
    namespace_index = preparation_source.index("create_v5_namespace_exclusive(")
    assert chain_index < gate_index < namespace_index
    assert chain_index < preparation_source.index("load_and_join_bound_inputs(")


def test_bound_identity_rejects_a_symlink_component_before_read(
    tmp_path: subject.Path,
) -> None:
    target = tmp_path / "physical.bin"
    payload = b"identity"
    target.write_bytes(payload)
    link = tmp_path / "linked.bin"
    _make_test_symlink(link, target)
    identity = subject.FileIdentity(
        "linked.bin", len(payload), subject.hashlib.sha256(payload).hexdigest()
    )
    with pytest.raises(subject.IdentityError, match="symlink or junction"):
        subject.revalidate_file_identity(tmp_path, identity)


def test_alias_and_transaction_reject_dangling_link_preexistence(
    tmp_path: subject.Path,
) -> None:
    source_payload = b"alias-source"
    source = tmp_path / "source.bin"
    source.write_bytes(source_payload)
    output_root = tmp_path / subject.OUTPUT_ROOT_RELATIVE
    output_root.mkdir(parents=True)
    stage_rel = subject.OUTPUT_ROOT_RELATIVE + "/.alias-stage"
    final_rel = subject.OUTPUT_ROOT_RELATIVE + "/alias-final.bin"
    stage = tmp_path / stage_rel
    _make_test_symlink(stage, tmp_path / "absent-stage-target")
    spec = subject.AliasSpec(
        subject.FileIdentity(
            "source.bin",
            len(source_payload),
            subject.hashlib.sha256(source_payload).hexdigest(),
        ),
        stage_rel,
        final_rel,
    )
    alias_authorization = subject.PredataAuthorization(
        valid_control(aliases_pass=False), artifact_root=tmp_path
    )
    with pytest.raises(subject.OutputPublicationError, match="already exists"):
        subject.publish_exact_byte_alias(
            tmp_path, spec, authorization=alias_authorization
        )
    stage.unlink()

    transaction = output_root / subject.TRANSACTION_DIRNAME
    _make_test_symlink(
        transaction,
        tmp_path / "absent-transaction-target",
        target_is_directory=True,
    )
    payloads = {
        basename: (
            b"synthetic",
            "PARQUET" if basename.endswith(".parquet") else "JSON",
            1,
            subject.hashlib.sha256(b"synthetic").hexdigest(),
        )
        for basename in subject.RUNNER_ARTIFACT_BASENAMES
    }
    with pytest.raises(subject.OutputPublicationError, match="already exists"):
        subject.stage_serialized_runner_artifacts(
            tmp_path,
            payloads,
            authorization=subject.PredataAuthorization(
                valid_control(), artifact_root=tmp_path
            ),
        )


def test_production_run_path_completes_synthetic_negative_without_any_fit(
    tmp_path: subject.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, _ = synthetic_output_prepared(tmp_path)
    site = prepared.joined.site.copy()
    for response in subject.RESPONSES:
        site[response] = np.zeros(subject.EXPECTED_SITE_ROWS, dtype=np.float64)
    prepared = subject.dataclasses.replace(
        prepared,
        joined=subject.dataclasses.replace(prepared.joined, site=site),
    )
    observation = zero_threadpool_evidence()["first_observation_by_phase"][
        "BEFORE_EXPLICIT_CONTEXT"
    ]["threadpools"]
    recorder = subject.ThreadpoolCaptureRecorder()
    recorder.capture(
        phase="BEFORE_EXPLICIT_CONTEXT",
        label="synthetic-before-no-fit",
        observed=observation,
    )
    prepared.authorization.threadpool_recorder = recorder
    prepared.authorization.artifact_root = tmp_path
    prepared.authorization.alias_files_published = True
    prepared.authorization.alias_metadata_validated = True
    output_root = tmp_path / subject.OUTPUT_ROOT_RELATIVE
    output_root.mkdir(parents=True)
    alias_identities: list[subject.FileIdentity] = []
    for spec in subject.ALIAS_SPECS:
        alias_path = tmp_path / spec.final_relative_path
        alias_path.write_bytes(b"synthetic-alias")
        alias_identities.append(
            subject.FileIdentity(
                spec.final_relative_path,
                len(b"synthetic-alias"),
                subject.hashlib.sha256(b"synthetic-alias").hexdigest(),
            )
        )
    prepared = subject.dataclasses.replace(
        prepared, alias_identities=tuple(alias_identities)
    )

    def synthetic_parquet_writer(
        path: subject.Path,
        *,
        schema: object,
        arrays: object,
        row_count: int,
    ) -> subject.StagedIdentity:
        del schema, arrays
        payload = b"synthetic-parquet-without-fit"
        size, digest = subject._exclusive_write_bytes(path, payload)
        return subject.StagedIdentity(
            path=path.as_posix(),
            size_bytes=size,
            sha256=digest,
            format="PARQUET",
            row_count=row_count,
            logical_sha256=subject.hashlib.sha256(
                b"synthetic-logical-without-fit"
            ).hexdigest(),
        )

    monkeypatch.setattr(
        subject, "write_parquet_exclusive", synthetic_parquet_writer
    )
    result = subject.run_prepared_execution(prepared)
    assert result.decision_document["status"] == (
        "STOP_NO_TARGET_FREE_INCREMENTAL_FAMILY"
    )
    assert result.decision_document["selected_family"] is None
    assert result.threadpool_evidence["capture_counts"] == {
        "BEFORE_EXPLICIT_CONTEXT": 1,
        "INSIDE_CONTEXT": 0,
        "AFTER_CONTEXT": 0,
    }
    assert len(result.staged_output_identities) == 7
    transaction = output_root / subject.TRANSACTION_DIRNAME
    ledger = subject.strict_json_loads(
        (transaction / "TARGET_FREE_FIT_LEDGER_V5.json").read_bytes()
    )
    assert ledger["executed_counts"]["completed_total_decision_units"] == 0
    assert ledger["skipped_counts"]["skipped_total_decision_units"] == 1332
    assert not any(
        subject._lexists(output_root / basename)
        for basename in (
            *subject.RUNNER_ARTIFACT_BASENAMES,
            *subject.SEALER_ARTIFACT_BASENAMES,
        )
    )


def test_exact7_staging_is_create_if_absent_and_never_publishes_final(
    tmp_path: subject.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = constant_raw_evaluations()
    derived = empty_derived_evaluation(raw)
    decision = subject.build_family_decision_document(raw, derived)
    ledger = subject.build_fit_ledger_document(
        raw, decision_status=str(decision["status"])
    )
    distribution = subject.build_distribution_document(
        raw, derived, decision_status=str(decision["status"])
    )
    bundle = subject.EvaluationBundle(
        raw_evaluations=raw,
        derived_evaluation=derived,
        decision_document=decision,
        fit_ledger_document=ledger,
        distribution_document=distribution,
        duplicate_metric_records=tuple(
            item
            for response in subject.RESPONSES
            for item in raw[response].duplicate_metric_records
        ),
        stability_records=tuple(
            [
                item
                for response in subject.RESPONSES
                for item in raw[response].stability_records
            ]
            + list(derived.stability_records)
        ),
    )
    output_root = tmp_path / subject.OUTPUT_ROOT_RELATIVE
    output_root.mkdir(parents=True)
    alias_identities = []
    for spec in subject.ALIAS_SPECS:
        path = tmp_path / spec.final_relative_path
        path.write_bytes(b"synthetic-alias")
        alias_identities.append(
            subject.FileIdentity(
                spec.final_relative_path,
                len(b"synthetic-alias"),
                subject.hashlib.sha256(b"synthetic-alias").hexdigest(),
            )
        )
    authorization = subject.PredataAuthorization(
        valid_control(),
        artifact_root=tmp_path,
        alias_files_published=True,
        alias_metadata_validated=True,
    )
    authorization.authorize_arrow_handoff()
    prepared = subject.PreparedExecution(
        artifact_root=tmp_path,
        attempt_id="target_free_duplicate_v5__20260811T123456123456Z",
        amendment={},
        amendment_v4={},
        amendment_v5={},
        v3_failure_incident={},
        bound_paths={},
        authorization=authorization,
        joined=None,
        runtime_evidence={},
        alias_identities=tuple(alias_identities),
    )

    monkeypatch.setattr(
        subject, "build_oof_arrow_table", lambda *_args: (object(), ())
    )
    monkeypatch.setattr(
        subject, "build_derived_arrow_table", lambda *_args: (object(), ())
    )

    def fake_parquet_writer(
        path: subject.Path,
        *,
        schema: object,
        arrays: object,
        row_count: int,
    ) -> subject.StagedIdentity:
        del schema, arrays
        payload = b"synthetic-parquet"
        size, digest = subject._exclusive_write_bytes(path, payload)
        return subject.StagedIdentity(
            path=path.as_posix(),
            size_bytes=size,
            sha256=digest,
            format="PARQUET",
            row_count=row_count,
            logical_sha256=subject.hashlib.sha256(
                b"synthetic-logical"
            ).hexdigest(),
        )

    monkeypatch.setattr(subject, "write_parquet_exclusive", fake_parquet_writer)
    staged = subject.stage_evaluation_bundle(prepared, bundle)
    assert len(staged) == 7
    assert [subject.Path(item.path).name for item in staged] == list(
        subject.RUNNER_ARTIFACT_BASENAMES
    )
    transaction = output_root / subject.TRANSACTION_DIRNAME
    assert {item.name for item in transaction.iterdir()} == set(
        subject.RUNNER_ARTIFACT_BASENAMES
    )
    assert not any(
        (output_root / basename).exists()
        for basename in subject.SEALER_ARTIFACT_BASENAMES
    )
    with pytest.raises(subject.OutputPublicationError, match="already exists"):
        subject.stage_evaluation_bundle(prepared, bundle)


def test_positive_synthetic_bundle_stages_exact7_without_any_fit(
    tmp_path: subject.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = completed_raw_evaluations()
    derived = completed_derived_evaluation(raw)
    decision = decision_document(
        [dict(raw[response].raw_result) for response in subject.RESPONSES],
        endpoints_pass=True,
    )
    assert decision["selected_family"] == "LOW_LEVEL_ISOBARIC_WIND_PROFILE"
    assert decision["family_clear_pass_results"] == {
        "LOW_LEVEL_ISOBARIC_WIND_PROFILE": True,
        "PBL_HEIGHT": True,
    }
    ledger = subject.build_fit_ledger_document(
        raw, decision_status=str(decision["status"])
    )
    assert ledger["executed_counts"] == {
        "completed_primary_model_units": 72,
        "completed_analytic_affine_units": 1260,
        "completed_total_decision_units": 1332,
    }
    distribution = subject.build_distribution_document(
        raw, derived, decision_status=str(decision["status"])
    )
    bundle = subject.EvaluationBundle(
        raw_evaluations=raw,
        derived_evaluation=derived,
        decision_document=decision,
        fit_ledger_document=ledger,
        distribution_document=distribution,
        duplicate_metric_records=tuple(
            item
            for response in subject.RESPONSES
            for item in raw[response].duplicate_metric_records
        ),
        stability_records=tuple(
            [
                item
                for response in subject.RESPONSES
                for item in raw[response].stability_records
            ]
            + list(derived.stability_records)
        ),
    )
    subject.validate_evaluation_bundle_consistency(bundle)
    short_derived = subject.dataclasses.replace(
        derived, stability_records=derived.stability_records[:-1]
    )
    with pytest.raises(subject.IntegrityError, match="fixed record counts"):
        subject.validate_evaluation_bundle_consistency(
            subject.dataclasses.replace(
                bundle,
                derived_evaluation=short_derived,
                stability_records=bundle.stability_records[:-1],
            )
        )
    broken_raw = dict(raw)
    first_response = subject.RESPONSES[0]
    broken_raw[first_response] = subject.dataclasses.replace(
        raw[first_response], independent_oof=None
    )
    with pytest.raises(
        subject.IntegrityError,
        match="independent selector lacks its exact four completed primary slots",
    ):
        subject.validate_evaluation_bundle_consistency(
            subject.dataclasses.replace(bundle, raw_evaluations=broken_raw)
        )

    inactive_raw = dict(raw)
    inactive_metrics = list(raw[first_response].duplicate_metric_records)
    inactive_metric = dict(inactive_metrics[0])
    inactive_metric["boundary_equality_flags"] = "YEAR_PSI_0.5"
    inactive_metrics[0] = inactive_metric
    inactive_raw[first_response] = subject.dataclasses.replace(
        raw[first_response], duplicate_metric_records=tuple(inactive_metrics)
    )
    inactive_bundle_metrics = tuple(
        item
        for response in subject.RESPONSES
        for item in inactive_raw[response].duplicate_metric_records
    )
    with pytest.raises(
        subject.IntegrityError, match="boundary/classification projection"
    ):
        subject.validate_evaluation_bundle_consistency(
            subject.dataclasses.replace(
                bundle,
                raw_evaluations=inactive_raw,
                duplicate_metric_records=inactive_bundle_metrics,
            )
        )

    output_root = tmp_path / subject.OUTPUT_ROOT_RELATIVE
    output_root.mkdir(parents=True)
    alias_identities = []
    for spec in subject.ALIAS_SPECS:
        path = tmp_path / spec.final_relative_path
        path.write_bytes(b"synthetic-alias")
        alias_identities.append(
            subject.FileIdentity(
                spec.final_relative_path,
                len(b"synthetic-alias"),
                subject.hashlib.sha256(b"synthetic-alias").hexdigest(),
            )
        )
    authorization = subject.PredataAuthorization(
        valid_control(),
        artifact_root=tmp_path,
        alias_files_published=True,
        alias_metadata_validated=True,
    )
    authorization.authorize_arrow_handoff()
    prepared = subject.PreparedExecution(
        artifact_root=tmp_path,
        attempt_id="target_free_duplicate_v5__20260811T123456123456Z",
        amendment={},
        amendment_v4={},
        amendment_v5={},
        v3_failure_incident={},
        bound_paths={},
        authorization=authorization,
        joined=None,
        runtime_evidence={},
        alias_identities=tuple(alias_identities),
    )
    monkeypatch.setattr(
        subject, "build_oof_arrow_table", lambda *_args: (object(), ())
    )
    monkeypatch.setattr(
        subject, "build_derived_arrow_table", lambda *_args: (object(), ())
    )

    def fake_parquet_writer(
        path: subject.Path,
        *,
        schema: object,
        arrays: object,
        row_count: int,
    ) -> subject.StagedIdentity:
        del schema, arrays
        size, digest = subject._exclusive_write_bytes(
            path, b"synthetic-parquet"
        )
        return subject.StagedIdentity(
            path=path.as_posix(),
            size_bytes=size,
            sha256=digest,
            format="PARQUET",
            row_count=row_count,
            logical_sha256=subject.hashlib.sha256(b"logical").hexdigest(),
        )

    monkeypatch.setattr(subject, "write_parquet_exclusive", fake_parquet_writer)
    staged = subject.stage_evaluation_bundle(prepared, bundle)
    assert len(staged) == 7
    assert [subject.Path(item.path).name for item in staged] == list(
        subject.RUNNER_ARTIFACT_BASENAMES
    )
    assert not any(
        subject._lexists(output_root / basename)
        for basename in subject.SEALER_ARTIFACT_BASENAMES
    )


def test_partial_staging_is_preserved_and_v5_retry_is_rejected(
    tmp_path: subject.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = constant_raw_evaluations()
    derived = empty_derived_evaluation(raw)
    decision = subject.build_family_decision_document(raw, derived)
    ledger = subject.build_fit_ledger_document(
        raw, decision_status=str(decision["status"])
    )
    distribution = subject.build_distribution_document(
        raw, derived, decision_status=str(decision["status"])
    )
    bundle = subject.EvaluationBundle(
        raw_evaluations=raw,
        derived_evaluation=derived,
        decision_document=decision,
        fit_ledger_document=ledger,
        distribution_document=distribution,
        duplicate_metric_records=tuple(
            item
            for response in subject.RESPONSES
            for item in raw[response].duplicate_metric_records
        ),
        stability_records=tuple(
            [
                item
                for response in subject.RESPONSES
                for item in raw[response].stability_records
            ]
            + list(derived.stability_records)
        ),
    )
    output_root = tmp_path / subject.OUTPUT_ROOT_RELATIVE
    output_root.mkdir(parents=True)
    alias_identities = []
    for spec in subject.ALIAS_SPECS:
        final = tmp_path / spec.final_relative_path
        final.write_bytes(b"synthetic-alias")
        alias_identities.append(
            subject.FileIdentity(
                spec.final_relative_path,
                len(b"synthetic-alias"),
                subject.hashlib.sha256(b"synthetic-alias").hexdigest(),
            )
        )
    authorization = subject.PredataAuthorization(
        valid_control(),
        artifact_root=tmp_path,
        alias_files_published=True,
        alias_metadata_validated=True,
    )
    authorization.authorize_arrow_handoff()
    prepared = subject.PreparedExecution(
        artifact_root=tmp_path,
        attempt_id="target_free_duplicate_v5__20260811T123456123456Z",
        amendment={},
        amendment_v4={},
        amendment_v5={},
        v3_failure_incident={},
        bound_paths={},
        authorization=authorization,
        joined=None,
        runtime_evidence={},
        alias_identities=tuple(alias_identities),
    )
    monkeypatch.setattr(
        subject, "build_oof_arrow_table", lambda *_args: (object(), ())
    )
    monkeypatch.setattr(
        subject, "build_derived_arrow_table", lambda *_args: (object(), ())
    )

    def fail_after_partial_parquet(
        path: subject.Path,
        *,
        schema: object,
        arrays: object,
        row_count: int,
    ) -> subject.StagedIdentity:
        del schema, arrays, row_count
        subject._exclusive_write_bytes(path, b"preserved-partial-parquet")
        raise subject.OutputPublicationError(
            "synthetic transaction failure; incident and V4 required"
        )

    monkeypatch.setattr(
        subject, "write_parquet_exclusive", fail_after_partial_parquet
    )
    with pytest.raises(subject.OutputPublicationError, match="incident and V4"):
        subject.stage_evaluation_bundle(prepared, bundle)
    transaction = output_root / subject.TRANSACTION_DIRNAME
    expected_partial = set(subject.RUNNER_ARTIFACT_BASENAMES[:4])
    assert {item.name for item in transaction.iterdir()} == expected_partial
    before = {
        item.name: item.read_bytes() for item in transaction.iterdir()
    }
    assert set(item.name for item in output_root.iterdir()) == {
        *(subject.Path(spec.final_relative_path).name for spec in subject.ALIAS_SPECS),
        subject.TRANSACTION_DIRNAME,
    }
    assert not any(
        subject._lexists(output_root / basename)
        for basename in (
            *subject.RUNNER_ARTIFACT_BASENAMES,
            *subject.SEALER_ARTIFACT_BASENAMES,
        )
    )
    with pytest.raises(subject.OutputPublicationError, match="already exists"):
        subject.stage_evaluation_bundle(prepared, bundle)
    assert {
        item.name: item.read_bytes() for item in transaction.iterdir()
    } == before


def test_main_wires_preparation_execution_and_pretty_stdout(
    tmp_path: subject.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    decision = decision_document()
    prepared_marker = object()
    observed: dict[str, object] = {}

    def fake_prepare(
        artifact_root: subject.Path,
        *,
        authorization_path: subject.Path,
        independent_go_path: subject.Path,
    ) -> object:
        observed["root"] = artifact_root
        observed["authorization"] = authorization_path
        observed["go"] = independent_go_path
        return prepared_marker

    identity_rows = (333, 348, 1, 352512, 1332, 624, 414720)
    identities = tuple(
        subject.StagedIdentity(
            path=(
                subject.OUTPUT_ROOT_RELATIVE
                + "/"
                + subject.TRANSACTION_DIRNAME
                + "/"
                + basename
            ),
            size_bytes=1,
            sha256="0" * 64,
            format=(
                "PARQUET"
                if basename.endswith(".parquet")
                else "CSV" if basename.endswith(".csv") else "JSON"
            ),
            row_count=row_count,
            logical_sha256="0" * 64,
        )
        for basename, row_count in zip(
            subject.RUNNER_ARTIFACT_BASENAMES, identity_rows, strict=True
        )
    )

    def fake_run(prepared: object) -> subject.RunnerResult:
        assert prepared is prepared_marker
        return subject.RunnerResult(
            attempt_id="target_free_duplicate_v5__20260811T123456123456Z",
            decision_document=decision,
            staged_output_identities=identities,
            threadpool_evidence=zero_threadpool_evidence(),
        )

    monkeypatch.setattr(subject, "prepare_execution", fake_prepare)
    monkeypatch.setattr(subject, "run_prepared_execution", fake_run)
    exit_code = subject.main(
        [
            "--root",
            str(tmp_path),
            "--authorization",
            str(tmp_path / "auth.json"),
            "--independent-go",
            str(tmp_path / "go.json"),
        ]
    )
    captured = capsysbinary.readouterr()
    assert exit_code == 0 and captured.err == b""
    assert captured.out.endswith(b"\n") and b"\r" not in captured.out
    parsed = subject.strict_json_loads(captured.out)
    assert parsed["status"] == (
        "RUNNER_STAGED_OUTPUTS_COMPLETE_PENDING_POSTRUN_AUDIT"
    )
    assert parsed["output_counts"]["runner_staged_outputs"] == 7
    assert observed["root"] == tmp_path.resolve()


@pytest.mark.parametrize("drift", [None, "plain_string", "reorder", "numeric"])
def test_v5_exact4_footer_gate_is_structural_and_precedes_root(
    tmp_path: subject.Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str | None,
) -> None:
    import pyarrow as pa

    class SyntheticParquetFile:
        def __init__(self, schema: object, rows: int, marker: str) -> None:
            self.schema_arrow = schema
            self.metadata = type("Metadata", (), {"num_rows": rows})()
            self.marker = marker
            self.closed = False

        def close(self) -> None:
            self.closed = True

    constructors = {
        "large_string": pa.large_string,
        "int16": pa.int16,
        "int64": pa.int64,
        "float64": pa.float64,
    }
    expected = subject._expected_footer_contracts()
    bound_inputs: dict[str, dict[str, object]] = {}
    bound_paths: dict[str, subject.Path] = {}
    contracts: dict[str, object] = {
        "common_footer_requirements": {},
        "forbidden_acceptance_routes": {},
        "gate_roles_exact_order": list(subject.FOOTER_GATE_ROLES),
        "structural_schema_canonicalization": {},
    }
    resources: dict[subject.Path, SyntheticParquetFile] = {}
    for role in subject.FOOTER_GATE_ROLES:
        columns, tokens, rows = expected[role]
        expected_types = [constructors[token]() for token in tokens]
        expected_schema = pa.schema(
            [pa.field(name, dtype) for name, dtype in zip(columns, expected_types, strict=True)]
        )
        observed_names = list(columns)
        observed_types = list(expected_types)
        if role == subject.FOOTER_GATE_ROLES[0]:
            if drift == "plain_string":
                observed_types[0] = pa.string()
            elif drift == "reorder":
                observed_names[0], observed_names[1] = observed_names[1], observed_names[0]
            elif drift == "numeric":
                first_float = tokens.index("float64")
                observed_types[first_float] = pa.float32()
        observed_schema = pa.schema(
            [
                pa.field(name, dtype)
                for name, dtype in zip(observed_names, observed_types, strict=True)
            ]
        )
        path = tmp_path / f"{role}.parquet"
        bound_paths[role] = path
        identity = {
            "path": f"synthetic/{role}.parquet",
            "size_bytes": 1,
            "sha256": "0" * 64,
        }
        bound_inputs[role] = identity
        witness = {
            "columns": list(expected_schema.names),
            "rows": rows,
            "types": [str(field.type) for field in expected_schema],
        }
        witness_bytes = subject.strict_json_dumps(witness, pretty=False)
        contracts[role] = {
            "bound_input_key": role,
            "column_names_exact": list(columns),
            "pyarrow_constructor_types_exact": list(tokens),
            "row_count_exact": rows,
            "source_identity": identity,
            "structural_schema_fingerprint": {
                "canonical_sha256": subject.hashlib.sha256(witness_bytes).hexdigest(),
                "canonical_size_bytes": len(witness_bytes),
            },
        }
        resources[path] = SyntheticParquetFile(observed_schema, rows, role)

    census = tmp_path / "manifest_census_v1.json"
    provenance = tmp_path / "RAW_RANGE_MANIFEST.parquet"
    bound_paths["census_manifest"] = census
    bound_paths["raw_range_manifest_parquet"] = provenance
    provenance_schema = pa.schema(
        [pa.field(name, pa.large_string()) for name in subject.PROVENANCE_REQUIRED_FIELDS]
    )
    resources[provenance] = SyntheticParquetFile(
        provenance_schema, 10_368, "provenance"
    )
    monkeypatch.setattr(subject, "validate_runner_zero_state", lambda _root: None)
    monkeypatch.setattr(subject, "validate_v3_failed_root_exact2", lambda _root: ())
    census_calls: list[subject.Path] = []
    monkeypatch.setattr(
        subject,
        "validate_census_alias_document",
        lambda _root, path: census_calls.append(path),
    )
    opened: list[SyntheticParquetFile] = []

    def fake_source_footer(
        _authorization: object,
        path: subject.Path,
        *_args: object,
        allowed_paths: object,
        **_kwargs: object,
    ) -> SyntheticParquetFile:
        assert path in tuple(allowed_paths)
        resource = resources[path]
        opened.append(resource)
        return resource

    monkeypatch.setattr(subject, "guarded_source_parquet_file", fake_source_footer)
    authorization = subject.PredataAuthorization(
        valid_control(
            bound_input_identity_and_schema_pass=False,
            aliases_pass=False,
        ),
        artifact_root=tmp_path,
        bound_input_identities_rehashed=True,
    )
    call = lambda: subject.validate_pre_root_source_metadata_gate(
        tmp_path,
        bound_paths,
        {"bound_inputs": bound_inputs},
        {"bound_parquet_footer_schemas_exact": contracts},
        authorization=authorization,
    )
    if drift is None:
        call()
        assert authorization.source_footer_metadata_validated is True
        assert authorization.validation.bound_input_identity_and_schema_pass is True
        assert census_calls == [census]
        assert len(opened) == 5 and all(item.closed for item in opened)
    else:
        with pytest.raises(subject.IntegrityError):
            call()
        assert authorization.source_footer_metadata_validated is False
        assert authorization.validation.bound_input_identity_and_schema_pass is False
        assert opened and all(item.closed for item in opened)
    assert authorization.source_metadata_scope_open is False
    assert authorization.arrow_handoff_started is False
    assert not (tmp_path / subject.OUTPUT_ROOT_RELATIVE).exists()


@pytest.mark.parametrize("mode", ["positive", "source_mutation", "source_hardlink"])
def test_v3_failed_root_exact2_rehashes_sources_and_proves_inode_separation(
    tmp_path: subject.Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    failed_root = tmp_path / subject.V3_FAILED_OUTPUT_ROOT_RELATIVE
    failed_root.mkdir(parents=True)
    sources = (
        tmp_path / "manifest_census_v1.json",
        tmp_path / "raw" / "RAW_RANGE_MANIFEST.parquet",
    )
    sources[1].parent.mkdir()
    aliases = tuple(
        tmp_path / identity.path for identity in subject.V3_FAILED_ALIAS_IDENTITIES
    )
    payloads = (b"census-source", b"provenance-source")
    for source, alias, payload in zip(sources, aliases, payloads, strict=True):
        source.write_bytes(payload)
        if mode == "source_hardlink":
            subject.os.link(source, alias)
        else:
            alias.write_bytes(payload)
    if mode == "source_mutation":
        sources[0].write_bytes(b"mutated")

    expected_by_name = {
        "manifest_census_v1.json": payloads[0],
        "RAW_RANGE_MANIFEST.parquet": payloads[1],
        "FIELD_CENSUS_LOCK.json": payloads[0],
        "PROVENANCE_LEDGER.parquet": payloads[1],
    }

    def fake_revalidate(root: subject.Path, identity: subject.FileIdentity) -> subject.Path:
        path = root / identity.path
        expected_payload = expected_by_name[path.name]
        if path.read_bytes() != expected_payload:
            raise subject.IdentityError("synthetic source identity differs")
        return path

    monkeypatch.setattr(subject, "revalidate_file_identity", fake_revalidate)
    if mode == "positive":
        incident_aliases = {}
        source_identities = (
            subject.FileIdentity("manifest_census_v1.json", 4_523, "0269f6717b1110fafb6bba052c78824c616c648dbbdfe9f630947012c3f571a1"),
            subject.FileIdentity("raw/RAW_RANGE_MANIFEST.parquet", 3_812_043, "11849682fd1c65ca20cfcb3002d3708beccf10ef862bb0f03d054c2c030790d5"),
        )
        for name, alias_identity, source_identity in zip(
            ("FIELD_CENSUS_LOCK", "PROVENANCE_LEDGER"),
            subject.V3_FAILED_ALIAS_IDENTITIES,
            source_identities,
            strict=True,
        ):
            incident_aliases[name] = {
                "alias_identity": subject.dataclasses.asdict(alias_identity),
                "source_identity": subject.dataclasses.asdict(source_identity),
                "source_and_alias_bytes_equal": True,
                "source_and_alias_file_ids_distinct": True,
                "source_and_alias_samefile": False,
                "source_persistent_link_count": 1,
                "alias_persistent_link_count": 1,
            }
        incident = {
            "postfailure_state": {
                "output_root": subject.V3_FAILED_OUTPUT_ROOT_RELATIVE + "/",
                "output_root_entry_count": 2,
                "output_root_entries_exact_order": [item.name for item in aliases],
                "transaction_root_present": False,
                "no_other_output_root_entries": True,
                "aliases": incident_aliases,
            }
        }
        assert len(subject.validate_v3_failed_root_exact2(tmp_path, incident)) == 2
    else:
        with pytest.raises((subject.IdentityError, subject.PredataAuthorizationError)):
            subject.validate_v3_failed_root_exact2(tmp_path)


def test_v5_inherited_numeric_ast_and_generation_whitelist_are_frozen() -> None:
    v3_path = subject.REPOSITORY_ROOT / "scripts/run_noaa_gfs_target_free_duplicate_v3.py"
    v5_path = subject.REPOSITORY_ROOT / "scripts/run_noaa_gfs_target_free_duplicate_v5.py"
    trees = {
        version: ast.parse(path.read_text(encoding="utf-8"))
        for version, path in (("v3", v3_path), ("v5", v5_path))
    }
    functions = {}
    for version, tree in trees.items():
        functions[version] = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    inherited = (
        "persist_float",
        "regression_metrics",
        "pearson_correlation",
        "spearman_correlation",
        "analytic_affine_fit_predict",
        "qualifying_affine_predictors",
        "classify_component",
        "evaluate_stability_gate",
        "monthly_iqr_stability",
        "exact_one_to_one_join",
        "assign_fold_ordinals",
        "capacity_aggregate_raw_uv",
        "manual_ridge_pipeline_predict",
        "serial_extra_trees_predict",
        "_validate_exact_crossfit_arrays",
        "crossfit_affine_response",
    )
    for name in inherited:
        assert ast.dump(functions["v5"][name], include_attributes=False) == ast.dump(
            functions["v3"][name], include_attributes=False
        )
    assert tuple(subject.RUNNER_ARTIFACT_BASENAMES) == (
        "TARGET_FREE_DUPLICATE_METRICS.csv",
        "TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE.csv",
        "TARGET_FREE_FAMILY_DECISION.json",
        "TARGET_FREE_DUPLICATE_OOF_V5.parquet",
        "TARGET_FREE_FIT_LEDGER_V5.json",
        "TARGET_FREE_DISTRIBUTION_STABILITY_V5.json",
        "TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V5.parquet",
    )
    assert all(
        path.replace("_v5.py", "_v3.py")
        in subject._expected_v4_authority_scope()["authorized_code_and_test_paths"]
        for path in subject.CODE_IDENTITY_FILES.values()
    )

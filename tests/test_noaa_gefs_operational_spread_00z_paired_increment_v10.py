from __future__ import annotations

import ast
import concurrent.futures
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "scripts/download_noaa_gefs_operational_spread_00z_original_v10.py"
RUNNER = ROOT / "scripts/run_noaa_gefs_operational_spread_00z_paired_increment_v10.py"
REGRESSION = ROOT / "scripts/audit_noaa_gefs_operational_spread_00z_v10_decoder_gate.py"
V7 = ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v7.json"
V8 = ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v8.json"
V9 = ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v9.json"
V10 = ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v10.json"
V7_SEAL = ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_original_v7_stage1_protocol_failure_seal_v1.json"
AUTHORIZATION = ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_original_v10_user_authorized_source_boundary_v1.json"


def load_extractor(name: str):
    spec = importlib.util.spec_from_file_location(name, EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v10_preregister_and_terminal_lineage_are_exact() -> None:
    v10 = json.loads(V10.read_text(encoding="utf-8"))
    assert v10["status"] == "FROZEN_V10_USER_AUTHORIZED_SOURCE_ONLY_COLD_DECODER_GATE_BEFORE_SOURCE_CHANGE_OR_ANY_V10_NETWORK_VALUE_LABEL_FIT_PREDICTION_SCORE"
    assert v10["supersedes"]["sha256"] == "8dd85674d3bb7a706d1e624267b55763caf1cad4e8c0005bddbd4d1003ae9f6d"
    assert v10["authorization_lineage"]["sha256"] == "0ef28e4d641b6a11291d974cbcd329a7d25ba3be0a47c46737e7a855b0aa0d68"
    assert v10["terminal_v7_lineage"]["sha256"] == "797eb8188f4f8726f1941efdfd194877a77c0018c1bd5a8ee22f958f60bdc7ba"
    assert v10["terminal_v7_lineage"]["v7_partial_or_preallocated_pack_reuse"] == 0
    science = v10["immutable_scientific_inheritance"]
    assert science["candidate_id"] == "gefs00z_0p50_10m_850_mean_spread_w025"
    assert science["blend_weight"] == 0.25
    assert science["ranges"] == 26280 and science["payload_bytes"] == 11524284739
    assert V7.is_file() and V8.is_file() and V9.is_file() and V7_SEAL.is_file() and AUTHORIZATION.is_file()


def test_decoder_uses_message_api_and_forbids_sample_warmup() -> None:
    source = EXTRACTOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    unlocked = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_decode_two_messages_unlocked")
    unlocked_attrs = {node.attr for node in ast.walk(unlocked) if isinstance(node, ast.Attribute)}
    assert "codes_new_from_message" in unlocked_attrs
    assert "codes_grib_new_from_file" not in unlocked_attrs
    assert "codes_grib_new_from_message" not in source
    assert "codes_grib_new_from_samples" not in source


def test_first_worker_completes_before_waiters_and_publishes_once(monkeypatch) -> None:
    module = load_extractor("_test_gefs_v10_gate_success")
    first_entered = threading.Event()
    release_first = threading.Event()
    call_lock = threading.Lock()
    calls = 0

    def fake(payload, plan):
        nonlocal calls
        with call_lock:
            calls += 1
            current = calls
        if current == 1:
            first_entered.set()
            assert release_first.wait(timeout=10)
        return ([{"call": current}], [float(current)])

    monkeypatch.setattr(module, "_decode_two_messages_unlocked", fake)
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(module.decode_two_messages, b"payload", object()) for _ in range(48)]
        assert first_entered.wait(timeout=10)
        threading.Event().wait(0.05)
        assert calls == 1
        assert module.decoder_gate_state() == {"ready": False, "failure": None, "first_success_count": 0}
        release_first.set()
        results = [future.result(timeout=10) for future in futures]
    assert len(results) == 48 and calls == 48
    assert module.decoder_gate_state() == {"ready": True, "failure": None, "first_success_count": 1}


def test_first_worker_failure_is_published_without_waiter_decode(monkeypatch) -> None:
    module = load_extractor("_test_gefs_v10_gate_failure")
    call_lock = threading.Lock()
    calls = 0

    def fail(payload, plan):
        nonlocal calls
        with call_lock:
            calls += 1
        raise ValueError("synthetic first-worker failure")

    monkeypatch.setattr(module, "_decode_two_messages_unlocked", fail)
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(module.decode_two_messages, b"payload", object()) for _ in range(48)]
        failures = []
        for future in futures:
            with pytest.raises((ValueError, RuntimeError)) as caught:
                future.result(timeout=10)
            failures.append(str(caught.value))
    assert calls == 1
    assert module.decoder_gate_state()["ready"] is True
    assert module.decoder_gate_state()["first_success_count"] == 0
    assert "synthetic first-worker failure" in module.decoder_gate_state()["failure"]
    assert len(failures) == 48


def test_regression_contract_is_fixed_24_by_240_and_offline() -> None:
    source = REGRESSION.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"THREADS", "TASKS"}
    }
    assert assignments == {"THREADS": 24, "TASKS": 240}
    assert "extractor.legacy.range_get = forbidden_range_get" in source
    assert "fixture_future_pack_or_model_reuse\": 0" in source
    assert "cold-unlocked" in source and "locked" in source


def test_no_v10_live_preflight_cli_and_fixture_is_v7_offline_only() -> None:
    module = load_extractor("_test_gefs_v10_no_live_preflight")
    source = EXTRACTOR.read_text(encoding="utf-8")
    assert "parser.add_argument(\"--execute-preflight\"" not in source
    assert "parser.add_argument(\"--freeze-preflight-lock\"" not in source
    assert "original_v7/preflight_one_range" in module.PREFLIGHT_OUTPUT.as_posix()
    assert "original_v10" in module.OUTPUT_ROOT.as_posix()
    assert module.PREFLIGHT_OUTPUT != module.OUTPUT_ROOT


def test_independent_review_gate_is_fail_closed(monkeypatch, tmp_path: Path) -> None:
    module = load_extractor("_test_gefs_v10_review")
    review = tmp_path / "review.json"
    monkeypatch.setattr(module, "INDEPENDENT_REVIEW", review)
    with pytest.raises(FileNotFoundError):
        module._review_identity()
    review.write_text(json.dumps({"verdict": "PASS", "v10_full_network_value_label_fit_score_access": 1}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="zero-access"):
        module._review_identity()


def test_frozen_plan_and_source_roots_are_unchanged() -> None:
    module = load_extractor("_test_gefs_v10_plan")
    plans, sizes = module.build_stage1_plan()
    assert len(plans) == 26280 and sum(sizes.values()) == 11524284739
    assert not any("gefs.2024" in plan.data_key or "gefs.2025" in plan.data_key for plan in plans)
    roots, recursive = module._source_roots_and_imports()
    root_names = {Path(record["path"]).name for record in roots}
    assert {
        EXTRACTOR.name,
        RUNNER.name,
        REGRESSION.name,
        "run_noaa_gefs_operational_spread_00z_paired_increment_v2.py",
    }.issubset(root_names)
    assert any(record["path"].endswith("src/metric.py") for record in recursive)


def test_runner_transform_remains_identifier_only() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "LEGACY_RUNNER_SHA" in source
    assert '"scientific_code_changes": 0' in source
    assert "FRESH_MODEL_OUTPUT" in source and "FRESH_MODEL_TOMBSTONE" in source
    assert "V7_FAILURE_SEAL" in source and "V10_AUTHORIZATION" in source
    tree = ast.parse(source)
    guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SystemExit"
        and any(isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "main" for inner in ast.walk(node))
        for node in ast.walk(guard)
    )


def test_runner_cli_static_audit_is_nonempty_parseable_pass() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", str(RUNNER), "--static-audit"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "PASS_ACTUAL_STAGE1_RUNNER_STATIC_CONTRACT_V10_FRESH_NAMESPACE"
    assert payload["scientific_code_changes"] == 0
    assert payload["config_v10_sha256"] == "b154793821dcebd9e859605006cbecf139c3753f4bab79693958bba90092e70d"


def test_runner_cli_unknown_argument_fails_nonzero() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", str(RUNNER), "--definitely-unknown-v11-argument"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "unrecognized arguments" in completed.stderr


def test_v10_namespace_and_zero_state() -> None:
    module = load_extractor("_test_gefs_v10_zero_state")
    assert "original_v10" in module.OUTPUT_ROOT.as_posix()
    assert "original_v7" not in module.OUTPUT_ROOT.as_posix()
    assert not module.OUTPUT_ROOT.exists()
    assert not module.STAGE2_ROOT.exists()
    assert not module.LAUNCH_LOCK.exists()
    assert not module.SOURCE_TOMBSTONE.exists()

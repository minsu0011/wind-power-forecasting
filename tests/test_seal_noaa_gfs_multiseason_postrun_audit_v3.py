from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import audit_noaa_gfs_multiseason_raw_postrun_v3 as auditor
from scripts import seal_noaa_gfs_multiseason_postrun_audit_v3 as subject


def write_bytes(path: Path, data: bytes = b"{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def identity(path: Path, root: Path | None = None) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix() if root is not None else str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def control_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    for relative in ("prereg", "independent_redteam", "raw", "decoded"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def fake_code_state(tmp_path: Path) -> dict[str, object]:
    paths: dict[str, Path] = {}
    for role in (
        "v2_auditor", "v2_auditor_test", "v3_auditor", "v3_auditor_test",
        "v3_sealer", "v3_sealer_test",
    ):
        path = tmp_path / f"{role}.py"
        names = auditor.V3_REQUIRED_TEST_NAMES if role == "v3_auditor_test" else ()
        source = "\n".join(f"def {name}():\n    pass\n" for name in names) or "VALUE = 1\n"
        write_bytes(path, source.encode("utf-8"))
        paths[role] = path
    return {
        "bound_identities": {role: identity(path) for role, path in paths.items()},
        "v2_recovery_support": {
            role: {"path": str(tmp_path / role), "size_bytes": 1, "sha256": "0" * 64}
            for role in auditor.V3_V2_SUPPORT_PATHS
        },
    }


def fake_inputs() -> dict[str, object]:
    return {
        "incident": {"path": "incident.json", "size_bytes": 1, "sha256": "0" * 64},
        "v2_recovery_controls": {
            role: dict(record) for role, record in auditor.V3_V2_CONTROL_IDENTITIES.items()
        },
        "recovery_transaction": dict(auditor.V3_TRANSACTION_IDENTITIES),
        "recovery_history": {
            "file_count": 3,
            "completion_locks": dict(auditor.V3_COMPLETION_LOCK_IDENTITIES),
            "postcommit_input_audit": dict(auditor.V3_POSTCOMMIT_INPUT_AUDIT_IDENTITY),
        },
        "recovery_progress": {
            "file_count": 104,
            "inventory": [],
            "inventory_canonical_sha256": auditor.V3_PROGRESS_INVENTORY_CANONICAL_SHA256,
            "final_checkpoint": dict(auditor.V3_FINAL_PROGRESS_IDENTITY),
        },
        "canonical_outputs": dict(auditor.V3_CANONICAL_OUTPUT_IDENTITIES),
        "original_v1_provenance": dict(auditor.V3_ORIGINAL_V1_PROVENANCE),
        "preaudit_zero_mutation_snapshot": {},
        "recovered_afterstate": {"closed": True},
    }


def test_sealer_source_is_static_network_zero_and_has_one_auth_publisher_call() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imported.isdisjoint({"requests", "urllib", "httpx", "aiohttp", "pandas", "pyarrow", "eccodes"})
    seal_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "seal"
    )
    publisher_calls = [
        node for node in ast.walk(seal_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "publish_json_no_overwrite"
    ]
    assert len(publisher_calls) == 1
    assert "publish_json_no_overwrite(control_paths(root)[\"independent_review\"]" not in source
    assert "publish_json_no_overwrite(control_paths(root)[\"independent_go\"]" not in source
    auditor_source = Path(auditor.__file__).read_text(encoding="utf-8")
    auditor_tree = ast.parse(auditor_source)
    auditor_imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(auditor_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert auditor_imports.isdisjoint({"socket", "requests", "urllib", "httpx", "aiohttp"})
    assert "if __name__ == \"__main__\":" in auditor_source


def test_contract_uses_exact_eight_support_roles_and_long_completion_lock_roles() -> None:
    assert set(auditor.V3_V2_SUPPORT_PATHS) == {
        "recovery_runner", "recovery_runner_test", "recovery_sealer",
        "recovery_sealer_test", "recovery_module", "recovery_bootstrap",
        "recovery_test", "recovery_bootstrap_test",
    }
    assert set(auditor.V3_COMPLETION_LOCK_IDENTITIES) == {
        "decode_mutex_complete", "raw_mutex_complete"
    }
    assert {
        role: Path(path).resolve()
        for role, path in auditor.V3_V2_SUPPORT_PATHS.items()
    } == {
        role: path.resolve() for role, path in subject.EXPECTED_V2_SUPPORT_PATHS.items()
    }


@pytest.mark.parametrize(
    "role",
    ("recovery_module", "recovery_bootstrap", "recovery_bootstrap_test"),
)
def test_support_alias_path_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    changed = dict(auditor.V3_V2_SUPPORT_PATHS)
    changed[role] = tmp_path / Path(changed[role]).name
    monkeypatch.setattr(auditor, "V3_V2_SUPPORT_PATHS", changed)
    with pytest.raises(subject.PostrunAuditSealError, match="canonical path mismatch"):
        subject.collect_code_state(subject.REPO)


def test_external_identity_rejects_relative_path_before_normalization() -> None:
    with pytest.raises(subject.PostrunAuditSealError, match="not absolute"):
        subject.strict_external_identity(Path("scripts/example.py"), label="relative")


def test_external_identity_rejects_symlink_before_resolve(tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    link = tmp_path / "alias.py"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink privilege unavailable: {exc}")
    with pytest.raises(subject.PostrunAuditSealError, match="symlink/junction"):
        subject.strict_external_identity(link.resolve(strict=False).parent / link.name, label="alias")


def test_preimport_auditor_guard_rejects_top_level_io(tmp_path: Path) -> None:
    source = tmp_path / "auditor.py"
    source.write_text("VALUE = open('secret.bin', 'rb')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unapproved import-time call"):
        subject._preimport_auditor_source_guard(source)


@pytest.mark.parametrize("placement", ("legacy", "mixed_case_cache_tag"))
def test_matching_auditor_bytecode_competitor_fails_closed(
    tmp_path: Path, placement: str
) -> None:
    source = tmp_path / "audit_example_v3.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    if placement == "legacy":
        candidate = source.with_suffix(".pyc")
    else:
        candidate = source.parent / "__pycache__" / "AuDiT_ExAmPlE_V3.CP313.PyC"
    write_bytes(candidate, b"poison")
    with pytest.raises(RuntimeError, match="bytecode/cache competitor"):
        subject._require_auditor_bytecode_absent(source)


@pytest.mark.parametrize(
    "role",
    (
        "v2_auditor", "v2_auditor_test", "v3_auditor", "v3_auditor_test",
        "v3_sealer", "v3_sealer_test",
    ),
)
def test_timestamp_valid_bytecode_is_rejected_for_every_executed_source_role(
    tmp_path: Path, role: str
) -> None:
    code = fake_code_state(tmp_path)
    source = Path(str(code["bound_identities"][role]["path"]))
    cache = source.parent / "__pycache__"
    cache.mkdir(exist_ok=True)
    candidate = cache / f"{source.stem}.cp313.pyc"
    py_compile.compile(str(source), cfile=str(candidate), doraise=True)
    with pytest.raises(
        subject.PostrunAuditSealError, match="executable-source bytecode/cache"
    ):
        subject._require_bound_bytecode_absent(code["bound_identities"])


@pytest.mark.parametrize("competitor", ("package", "mixed_pyd"))
def test_auditor_same_stem_package_or_extension_competitor_fails(
    tmp_path: Path, competitor: str
) -> None:
    source = tmp_path / "audit_example_v3.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    if competitor == "package":
        (tmp_path / "audit_example_v3").mkdir()
    else:
        write_bytes(tmp_path / "AuDiT_ExAmPlE_V3.CP313.PyD", b"extension")
    with pytest.raises(RuntimeError, match="source-origin competitor"):
        subject._require_auditor_source_competitors_absent(source)


def test_collect_progress_closes_exact_104_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    directory = root / "decoded" / "recovery_progress"
    directory.mkdir()
    for count in range(1, 105):
        write_bytes(
            directory
            / f"decoded_messages__{auditor.V3_RECOVERY_ATTEMPT_ID}__{count:06d}.json",
            f"{{\"count\":{count}}}".encode(),
        )
    inventory = [identity(path, root) for path in sorted(directory.iterdir())]
    encoded = subject.canonical_bytes(inventory)
    monkeypatch.setattr(auditor, "V3_PROGRESS_INVENTORY_CANONICAL_BYTES", len(encoded))
    monkeypatch.setattr(
        auditor, "V3_PROGRESS_INVENTORY_CANONICAL_SHA256", hashlib.sha256(encoded).hexdigest()
    )
    monkeypatch.setattr(auditor, "V3_FINAL_PROGRESS_IDENTITY", inventory[-1])
    declaration = subject.collect_progress(root)
    assert declaration["file_count"] == 104
    assert declaration["inventory"] == inventory
    write_bytes(directory / "junk.json")
    with pytest.raises(subject.PostrunAuditSealError, match="exact inventory"):
        subject.collect_progress(root)


def test_root_identity_map_tamper_fails_closed(tmp_path: Path) -> None:
    root = control_root(tmp_path)
    path = root / "prelaunch" / "control.json"
    write_bytes(path, b"sealed")
    expected = {"control": identity(path, root)}
    assert subject._verify_root_map(root, expected, label="control") == expected
    path.write_bytes(b"tampered")
    with pytest.raises(subject.PostrunAuditSealError, match="identity mismatch"):
        subject._verify_root_map(root, expected, label="control")


def test_zero_snapshot_and_control_temp_are_fail_closed(tmp_path: Path) -> None:
    root = control_root(tmp_path)
    snapshot = subject.build_zero_snapshot(root)
    assert snapshot["network_requests"] == 0
    temporary = (root / subject.AUTH_RELATIVE).with_name(
        f".{Path(subject.AUTH_RELATIVE).name}.tmp.1.deadbeef"
    )
    write_bytes(temporary)
    with pytest.raises(subject.PostrunAuditSealError, match="temporary"):
        subject.build_zero_snapshot(root)


def test_build_authorization_exact_schema_and_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    monkeypatch.setattr(subject, "validate_authorization_payload", lambda *_args: None)
    payload = subject.build_authorization(
        root,
        code,
        inputs,
        {"created_utc": "2026-08-11T00:00:00.000000Z"},
        created_utc="2026-08-11T00:00:01.000000Z",
    )
    assert set(payload) == auditor.V3_AUTH_KEYS
    assert payload["status"] == "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V3"
    assert payload["v2_recovery_support"] == code["v2_recovery_support"]
    assert payload["recovery_history"]["completion_locks"] == (
        auditor.V3_COMPLETION_LOCK_IDENTITIES
    )
    assert payload["required_command"] == auditor._v3_required_command(
        root, root / subject.AUTH_RELATIVE, root / subject.GO_RELATIVE
    )


@pytest.mark.parametrize("mutation", ("missing", "extra", "status", "attempt"))
def test_authorization_schema_or_header_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    payload = {key: None for key in auditor.V3_AUTH_KEYS}
    payload.update(
        {
            "schema_version": 3,
            "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V3",
            "status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V3",
            "created_utc": "2026-08-11T00:00:01.000000Z",
            "audit_attempt_id": "postrun_audit_v3__20260811T000001000000Z",
        }
    )
    monkeypatch.setattr(auditor, "_v3_validate_policy", lambda *_args, **_kwargs: None)
    if mutation == "missing":
        payload.pop("incident")
    elif mutation == "extra":
        payload["unknown"] = True
    elif mutation == "status":
        payload["status"] = "PASS"
    else:
        payload["audit_attempt_id"] = "postrun_audit_v3__wildcard"
    with pytest.raises(subject.PostrunAuditSealError, match="schema|header"):
        subject.validate_authorization_payload(tmp_path, payload, code, inputs)


@pytest.mark.parametrize(
    "mutation, expected_reason",
    (
        ("superseded_v2_auditor", "source identity binding"),
        ("superseded_v2_auditor_test", "source identity binding"),
        ("v3_auditor", "source identity binding"),
        ("v3_auditor_test", "source identity binding"),
        ("v3_sealer", "source identity binding"),
        ("v3_sealer_test", "source identity binding"),
        ("recovery_attempt_id", "recovery-attempt binding"),
        ("runtime_identity_sha256", "runtime-identity binding"),
        ("required_command", "required-command binding"),
        ("independent_review_required", "review/GO requirement"),
        ("independent_go_required", "review/GO requirement"),
        ("created_attempt", "created-time/attempt binding"),
    ),
)
def test_authorization_top_level_binding_family_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_reason: str,
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    original_validator = subject.validate_authorization_payload
    monkeypatch.setattr(subject, "validate_authorization_payload", lambda *_args: None)
    payload = subject.build_authorization(
        root,
        code,
        inputs,
        {"created_utc": "2026-08-11T00:00:00.000000Z"},
        created_utc="2026-08-11T00:00:01.000000Z",
    )
    monkeypatch.setattr(subject, "validate_authorization_payload", original_validator)
    monkeypatch.setattr(auditor, "_v3_validate_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        auditor,
        "_v3_recovered_afterstate_from_authorization",
        lambda _payload: inputs["recovered_afterstate"],
    )
    monkeypatch.setattr(auditor, "_v3_validate_zero_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(auditor, "_v3_validate_test_evidence", lambda *_args, **_kwargs: None)
    original_validator(root, payload, code, inputs)
    if mutation in {
        "superseded_v2_auditor", "superseded_v2_auditor_test", "v3_auditor",
        "v3_auditor_test", "v3_sealer", "v3_sealer_test",
    }:
        payload[mutation] = {**payload[mutation], "sha256": "f" * 64}
    elif mutation == "recovery_attempt_id":
        payload[mutation] = "decode_recovery_v2__wrong"
    elif mutation == "runtime_identity_sha256":
        payload[mutation] = "f" * 64
    elif mutation == "required_command":
        payload[mutation] = ["python", "audit.py"]
    elif mutation in {"independent_review_required", "independent_go_required"}:
        payload[mutation] = False
    else:
        payload["created_utc"] = "2026-08-11T00:00:02.000000Z"
    with pytest.raises(subject.PostrunAuditSealError, match=expected_reason):
        subject.validate_authorization_payload(root, payload, code, inputs)


def test_isolated_test_evidence_uses_three_exact_guarded_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = fake_code_state(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    monkeypatch.setattr(subject, "collect_code_state", lambda _root: code)

    def completed(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="1 passed in 0.01s\n", stderr="")

    monkeypatch.setattr(subject.subprocess, "run", completed)
    evidence = subject.run_immutable_test_evidence(tmp_path, tmp_path, code)
    assert calls == [auditor._v3_test_command(path) for path in subject.TEST_RELATIVES.values()]
    assert evidence["all_exit_codes_zero"] is True
    assert evidence["required_test_names"] == list(auditor.V3_REQUIRED_TEST_NAMES)
    assert set(evidence) == auditor.V3_TEST_EVIDENCE_KEYS


def test_isolated_test_failure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = fake_code_state(tmp_path)
    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    monkeypatch.setattr(subject, "collect_code_state", lambda _root: code)
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="1 failed in 0.01s\n", stderr="boom"
        ),
    )
    with pytest.raises(subject.PostrunAuditSealError, match="isolated test failed"):
        subject.run_immutable_test_evidence(tmp_path, tmp_path, code)


def test_publish_json_no_overwrite_preserves_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "authorization.json"
    destination.write_bytes(b"existing")
    with pytest.raises(subject.PostrunAuditSealError, match="already exists"):
        subject.publish_json_no_overwrite(destination, {"new": True})
    assert destination.read_bytes() == b"existing"


def test_publish_json_race_never_overwrites_attacker_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "authorization.json"

    def racing_link(_source: Path, target: Path, **_kwargs: object) -> None:
        Path(target).write_bytes(b"attacker")
        raise FileExistsError("race")

    monkeypatch.setattr(subject.os, "link", racing_link)
    with pytest.raises(subject.PostrunAuditSealError, match="create-if-absent"):
        subject.publish_json_no_overwrite(destination, {"authorized": True})
    assert destination.read_bytes() == b"attacker"
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_seal_publishes_exactly_one_authorization_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    code = {"bound_identities": {}, "v2_recovery_support": {}}
    inputs = {"stable": True}
    evidence = {"tests": "pass"}
    payload = {"schema_version": 3, "artifact_type": "TEST_AUTH"}
    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    monkeypatch.setattr(subject, "install_entry_guard", lambda: None)
    monkeypatch.setattr(subject, "collect_code_state", lambda _root: code)
    monkeypatch.setattr(subject, "collect_root_inputs", lambda _root, _code: inputs)
    monkeypatch.setattr(
        subject,
        "run_immutable_test_evidence",
        lambda _root, _workspace, _code: evidence,
    )
    monkeypatch.setattr(
        subject,
        "build_authorization",
        lambda *_args, **_kwargs: payload,
    )
    calls: list[Path] = []
    original = subject.publish_json_no_overwrite

    def spy(path: Path, value: dict[str, object]) -> None:
        calls.append(path)
        original(path, value)

    monkeypatch.setattr(subject, "publish_json_no_overwrite", spy)
    report = subject.seal(root, tmp_path)
    assert report["authorization_files_published"] == 1
    assert calls == [root / subject.AUTH_RELATIVE]
    assert (root / subject.AUTH_RELATIVE).is_file()
    assert not os.path.lexists(root / subject.REVIEW_RELATIVE)
    assert not os.path.lexists(root / subject.GO_RELATIVE)


def test_postpublication_failure_reports_one_and_never_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = control_root(tmp_path)
    code = {"bound_identities": {}, "v2_recovery_support": {}}
    inputs = {"stable": True}
    payload = {"schema_version": 3, "artifact_type": "TEST_AUTH"}
    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    monkeypatch.setattr(subject, "install_entry_guard", lambda: None)
    postpublication = {"code_rechecked": False, "afterstate_rechecked": False}

    def code_state(_root: Path) -> dict[str, object]:
        if os.path.lexists(root / subject.AUTH_RELATIVE):
            postpublication["code_rechecked"] = True
        return code

    def root_inputs(_root: Path, _code: dict[str, object]) -> dict[str, object]:
        if os.path.lexists(root / subject.AUTH_RELATIVE):
            postpublication["afterstate_rechecked"] = True
            raise subject.PostrunAuditSealError(
                "forced postpublication afterstate failure"
            )
        return inputs

    monkeypatch.setattr(subject, "collect_code_state", code_state)
    monkeypatch.setattr(subject, "collect_root_inputs", root_inputs)
    monkeypatch.setattr(
        subject,
        "run_immutable_test_evidence",
        lambda _root, _workspace, _code: {"tests": "pass"},
    )
    monkeypatch.setattr(
        subject, "build_authorization", lambda *_args, **_kwargs: payload
    )
    rc = subject.main(["--root", str(root)])
    report = json.loads(capsys.readouterr().out)
    destination = root / subject.AUTH_RELATIVE
    assert rc == 3
    assert destination.is_file()
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert report["status"] == "FAIL_V3_POSTRUN_AUDIT_AUTHORIZATION_POSTPUBLICATION"
    assert report["authorization_files_published"] == 1
    assert report["authorization_rollback_attempted"] is False
    assert report["authorization"] == identity(destination, root)
    assert postpublication == {
        "code_rechecked": True,
        "afterstate_rechecked": True,
    }


def test_seal_existing_authorization_fails_before_evidence_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    destination = root / subject.AUTH_RELATIVE
    write_bytes(destination, b"immutable")
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("preflight/evidence must not run")

    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    monkeypatch.setattr(subject, "install_entry_guard", lambda: None)
    monkeypatch.setattr(subject, "collect_code_state", forbidden)
    with pytest.raises(subject.PostrunAuditSealError, match="already exists"):
        subject.seal(root, tmp_path)
    assert called is False
    assert destination.read_bytes() == b"immutable"


def test_imported_programmatic_seal_fails_before_root_read_or_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("root read forbidden before entrypoint gate")

    monkeypatch.setattr(subject, "require_prepublication_control_state", forbidden)
    with pytest.raises(subject.PostrunAuditSealError, match="canonical"):
        subject.seal(root, tmp_path)
    assert called is False
    assert not os.path.lexists(root / subject.AUTH_RELATIVE)


def test_direct_file_write_entrypoint_is_rejected_without_publication(tmp_path: Path) -> None:
    root = control_root(tmp_path)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPYCACHEPREFIX", None)
    completed = subprocess.run(
        [sys.executable, "-B", str(Path(subject.__file__).resolve()), "--root", str(root)],
        cwd=subject.REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    report = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert report["status"] == "FAIL_V3_POSTRUN_AUDIT_AUTHORIZATION_SEAL"
    assert "canonical" in report["error"]
    assert not os.path.lexists(root / subject.AUTH_RELATIVE)


def test_canonical_module_without_dash_b_is_rejected_before_publication(tmp_path: Path) -> None:
    root = control_root(tmp_path)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPYCACHEPREFIX", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            subject.CANONICAL_MODULE_ENTRYPOINT,
            "--root",
            str(root.resolve()),
        ],
        cwd=subject.REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    report = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert report["status"] == "FAIL_V3_POSTRUN_AUDIT_AUTHORIZATION_SEAL"
    assert "orig_argv" in report["error"]
    assert not os.path.lexists(root / subject.AUTH_RELATIVE)


def test_pycache_prefix_is_rejected_before_auditor_import(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPYCACHEPREFIX"] = str(tmp_path / "poison-cache")
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import scripts.seal_noaa_gfs_multiseason_postrun_audit_v3",
        ],
        cwd=subject.REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "no pycache prefix" in completed.stderr


def test_network_route_is_unconditionally_denied() -> None:
    with pytest.raises(subject._NetworkDenied, match="network access is forbidden"):
        subject._network_forbidden(("127.0.0.1", 1))

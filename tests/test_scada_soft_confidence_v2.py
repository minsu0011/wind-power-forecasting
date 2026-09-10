from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.run_scada_soft_confidence_v2_code_only import (
    FROZEN_IDENTITIES,
    content_record,
    file_record,
    validate_identity_closure,
    write_new_utf8,
)
from src.scada_soft_confidence_v2 import (
    RAW_WEIGHT_FLOOR,
    continuous_soft_confidence_weights,
    scientific_prerequisite_decision,
)


def test_exact_formula_and_positive_floor() -> None:
    result = continuous_soft_confidence_weights(
        np.array([1.0, 0.8, np.nan]),
        np.array([1.0, 0.5, 1.0]),
        np.array([0.0, 0.25, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(result.confidence, [1.0, 0.3, 0.0])
    np.testing.assert_allclose(result.raw_weight, [1.0, 0.65, 0.5])
    assert RAW_WEIGHT_FLOOR == 0.5
    assert np.all(result.normalized_weight > 0)
    assert np.isclose(result.normalized_weight.mean(), 1.0)


def test_all_eligible_rows_are_retained_even_when_proxy_missing() -> None:
    size = 75
    result = continuous_soft_confidence_weights(
        np.full(size, np.nan), np.zeros(size), np.ones(size), np.full(size, np.nan)
    )
    assert len(result.normalized_weight) == size
    np.testing.assert_array_equal(result.normalized_weight, np.ones(size))


def test_shape_and_empty_guards_are_not_support_thresholds() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        continuous_soft_confidence_weights(*(np.array([]) for _ in range(4)))
    with pytest.raises(ValueError, match="identical shape"):
        continuous_soft_confidence_weights(
            np.ones(2), np.ones(1), np.ones(2), np.ones(2)
        )


def test_scientific_prerequisite_requires_all_groups() -> None:
    assert scientific_prerequisite_decision({"G1": True, "G2": True, "G3": False}) == (
        "TRACK_B_V2_STOP_INVALID_PROXY_PREREQUISITE"
    )
    assert scientific_prerequisite_decision({"G1": True, "G2": True, "G3": True}) == (
        "TRACK_B_V2_AWAIT_INDEPENDENT_PRELAUNCH_GO"
    )
    with pytest.raises(ValueError, match="exactly"):
        scientific_prerequisite_decision({"G1": True, "G2": True})


def test_frozen_dependency_hashes_are_exact() -> None:
    for path, (size, digest) in FROZEN_IDENTITIES.items():
        assert path.stat().st_size == size
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_identity_closure_rejects_duplicate_role_or_path(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("x=1\n", encoding="utf-8")
    second.write_text("x=2\n", encoding="utf-8")
    a = file_record(first, "runner")
    b = file_record(second, "runner")
    with pytest.raises(RuntimeError, match="role"):
        validate_identity_closure([a, b])
    b["role"] = "module"
    b["path"] = a["path"]
    with pytest.raises(RuntimeError, match="path"):
        validate_identity_closure([a, b])


def test_runner_is_code_only_and_has_no_hidden_normal_support_floor() -> None:
    source = Path("scripts/run_scada_soft_confidence_v2_code_only.py").read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "read_csv" not in source
    assert "read_parquet" not in source
    assert "sample_submission" not in source
    assert "minimum_normal_rows\": None" in source
    assert "hidden_minimum_row_thresholds\": 0" in source
    assert source.index("verify_frozen()") < source.index(
        "from src.scada_soft_confidence_v2 import scientific_prerequisite_decision"
    )


def test_actual_main_path_runtime_smoke_reaches_terminal_payload() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_scada_soft_confidence_v2_code_only.py", "--runtime-smoke"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "RUNTIME_SMOKE_PASS" in completed.stdout
    assert "TRACK_B_V2_STOP_INVALID_PROXY_PREREQUISITE" in completed.stdout


def test_exact_utf8_writer_preserves_lf_bytes_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "exact.json"
    content = "{\n  \"line\": true\n}\n"
    expected = content_record(path, "test", content)
    write_new_utf8(path, content)
    assert path.read_bytes() == content.encode("utf-8")
    assert file_record(path, "test") == expected
    with pytest.raises(FileExistsError):
        write_new_utf8(path, content)

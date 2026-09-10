from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts import run_feature_exit_nonwind_cap025143_scale097_overlay as runner
from src.density_ratio import assemble_locked_group
from src.metric import CAPACITY_KWH, TARGET_COLS


COMPONENTS = (
    "lgb_l1",
    "lgb_q07",
    "shared_l1",
    "shared_q07",
    "top200_q07",
    "energy_q06",
)


def _recipe(*, g2_offsets: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)) -> dict[str, Any]:
    weights = {
        group: {name: float(name == "lgb_q07") for name in COMPONENTS}
        for group in TARGET_COLS
    }
    return {
        "ensemble": {
            "weights": weights,
            "affine": {
                group: {"scale": 1.0, "bias_kwh": 0.0}
                for group in TARGET_COLS
            },
            "clip": {
                group: {
                    "lower_capacity_fraction": 0.0,
                    "upper_capacity_fraction": 1.02,
                }
                for group in TARGET_COLS
            },
            "power_bins": {
                "kpx_group_1": None,
                "kpx_group_2": {
                    "edges_cf": [0.0, 0.25, 0.50, 0.75, 1.02],
                    "delta_kwh": list(g2_offsets),
                },
                "kpx_group_3": None,
            },
        }
    }


def _component_frames(
    q07_cf: dict[str, list[float]], recipe: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    length = len(next(iter(q07_cf.values())))
    index = pd.date_range("2024-01-01 01:00", periods=length, freq="h", name="forecast_kst_dtm")
    frames = {
        name: pd.DataFrame(0.0, index=index, columns=TARGET_COLS)
        for name in COMPONENTS
    }
    for group in TARGET_COLS:
        frames["lgb_q07"][group] = np.asarray(q07_cf[group]) * CAPACITY_KWH[group]
    reference = pd.DataFrame(index=index, columns=TARGET_COLS, dtype=float)
    for group in TARGET_COLS:
        arrays = {name: frame[group].to_numpy(float) for name, frame in frames.items()}
        reference[group] = assemble_locked_group(
            arrays,
            group=group,
            capacity_kwh=CAPACITY_KWH[group],
            ensemble=recipe["ensemble"],
        )
    return frames, reference


def _paired_cf(index: pd.DatetimeIndex, values: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(values, index=index, columns=TARGET_COLS, dtype=float)


def test_science_triple_hashes_formula_and_effective_reference() -> None:
    expected = (
        (15_828, "91fa72355bab5bef80dcb73816e9bea162bf80ef26183333f9798d32770e42f9"),
        (2_110, "802dc38d6692eb6ecc546bc42743609e06d4bef79deaa6103dd5fba9843cb0da"),
        (2_704, "09e2b3ca41b8be49c5084b986dc6f0b2f7eedc432f65525ceb65cf2e3b3ac650"),
    )
    payloads = []
    for (path, size, digest), identity in zip(runner.SCIENCE_FILES, expected, strict=True):
        raw = path.read_bytes()
        assert (size, digest) == identity
        assert len(raw) == size
        assert hashlib.sha256(raw).hexdigest() == digest
        payloads.append(json.loads(raw))
    v1, v2, v3 = payloads
    assert runner.CAP_CF == float.fromhex("0x1.9bf2501bad320p-6")
    assert v1["delta_transfer_formula"]["cap_applied_exactly_once"] is True
    assert v1["delta_transfer_formula"]["per_group"][-1] == (
        "q1_kwh = capacity_kwh * clip(canonical_q07_kwh/capacity_kwh + "
        "capped_delta_cf, 0.0, 1.02)"
    )
    wrong = v2["correction"]["incorrect_v1_stress_reference"]
    assert wrong["sha256"] == "9ff992c2d3fa3ce0725600fda415f98bd9a9b114cf6cb259df6f0b796005b76a"
    assert wrong["must_not_be_used_by_this_experiment"] is True
    assert runner.GATE_REFERENCE[2] == "1c0a2e70996566c766a90a557c83ab1ba010277c31d202a1e4209188b94d112c"
    assert v2["correction"]["effective_original_gate_A0_reference"]["sha256"] == runner.GATE_REFERENCE[2]
    assert runner.GATE_RECIPE == (
        runner.ROOT / "artifacts/gate/v3/gate_recipe_snapshot.json",
        5_613,
        "c26e0b0b4268ed3780d85e1c3af5db38253d42e8a4230fbd910691f2ec6acd19",
    )
    assert runner.FINAL_RECIPE == (
        runner.ROOT / "artifacts/final_v3/v3_locked_full_2025__recipe.json",
        5_613,
        "c26e0b0b4268ed3780d85e1c3af5db38253d42e8a4230fbd910691f2ec6acd19",
    )
    assert v3["effective_science_head"] is True


def test_official_cache_lineage_binds_info_xlsx_before_source_lock() -> None:
    expected = (
        Path(r"data/local/open/info.xlsx"),
        3_823_422,
        "89e83a52e0eb2ce367a3573a96d6795ed4b4d4ac624965cb3530beec0cbd2bd6",
    )
    assert runner.INFO_XLSX == expected
    assert expected in runner.SMALL_PROVENANCE
    source = inspect.getsource(runner.run)
    assert source.index("SMALL_PROVENANCE") < source.index('"source_lock.json"')


def test_cap_is_applied_once_then_q1_receives_outer_component_clip() -> None:
    recipe = _recipe()
    q07 = {group: [1.01, 0.01] for group in TARGET_COLS}
    frames, reference = _component_frames(q07, recipe)
    index = reference.index
    control = _paired_cf(index, {group: [0.0, 0.0] for group in TARGET_COLS})
    exited = _paired_cf(index, {group: [0.10, -0.10] for group in TARGET_COLS})

    _, q1, _, _, audit = runner._assemble_delta(frames, control, exited, recipe, reference)

    for group in TARGET_COLS:
        capacity = CAPACITY_KWH[group]
        np.testing.assert_array_equal(q1[group].to_numpy(), [1.02 * capacity, 0.0])
        assert audit[group]["max_abs_capped_delta_cf"] == runner.CAP_CF
    source = inspect.getsource(runner._assemble_delta)
    assert source.count("np.clip(raw_dcf, -CAP_CF, CAP_CF)") == 1


def test_g2_uses_side_right_and_freezes_baseline_bin_after_crossing() -> None:
    offsets = (-400.0, -1000.0, 1200.0, -200.0)
    recipe = _recipe(g2_offsets=offsets)
    q07 = {
        "kpx_group_1": [0.30, 0.30],
        "kpx_group_2": [0.25, 0.49],
        "kpx_group_3": [0.30, 0.30],
    }
    frames, reference = _component_frames(q07, recipe)
    index = reference.index
    control = _paired_cf(index, {group: [0.0, 0.0] for group in TARGET_COLS})
    exited = _paired_cf(
        index,
        {
            "kpx_group_1": [0.0, 0.0],
            "kpx_group_2": [-0.01, runner.CAP_CF],
            "kpx_group_3": [0.0, 0.0],
        },
    )

    delta, q1, _, _, audit = runner._assemble_delta(frames, control, exited, recipe, reference)

    capacity = CAPACITY_KWH["kpx_group_2"]
    z1 = q1["kpx_group_2"].to_numpy()
    # At exactly 0.25, side="right" assigns the baseline to bin 1 (-1000).
    expected_a1 = np.clip(z1 + np.asarray([-1000.0, -1000.0]), 0.0, 1.02 * capacity)
    observed_a1 = reference["kpx_group_2"].to_numpy() + delta["kpx_group_2"].to_numpy()
    np.testing.assert_array_equal(observed_a1, expected_a1)
    assert audit["kpx_group_2"]["hypothetical_g2_bin_crossings"] == 2
    hypothetical_rebin_row_2 = np.clip(z1[1] + 1200.0, 0.0, 1.02 * capacity)
    assert observed_a1[1] != hypothetical_rebin_row_2


def test_a0_reference_tolerance_is_exactly_one_e_minus_nine() -> None:
    recipe = _recipe()
    q07 = {group: [0.30, 0.40] for group in TARGET_COLS}
    frames, reference = _component_frames(q07, recipe)
    control = _paired_cf(reference.index, {group: [0.0, 0.0] for group in TARGET_COLS})
    exited = control.copy()

    within = reference.copy()
    within.iloc[0, 0] += 5e-10
    runner._assemble_delta(frames, control, exited, recipe, within)

    outside = reference.copy()
    outside.iloc[0, 0] += 2e-9
    with pytest.raises(AssertionError, match="A0 reference reconstruction differs"):
        runner._assemble_delta(frames, control, exited, recipe, outside)


def test_control_vs_canonical_diagnostic_is_report_only_and_cannot_change_delta() -> None:
    recipe = _recipe()
    q07 = {group: [0.30, 0.40] for group in TARGET_COLS}
    frames, reference = _component_frames(q07, recipe)
    index = reference.index
    control_a = _paired_cf(index, {group: [0.10, 0.20] for group in TARGET_COLS})
    exit_a = _paired_cf(index, {group: [0.11, 0.21] for group in TARGET_COLS})
    control_b = _paired_cf(index, {group: [0.60, 0.70] for group in TARGET_COLS})
    exit_b = _paired_cf(index, {group: [0.61, 0.71] for group in TARGET_COLS})

    delta_a, q1_a, _, _, audit_a = runner._assemble_delta(
        frames, control_a, exit_a, recipe, reference
    )
    delta_b, q1_b, _, _, audit_b = runner._assemble_delta(
        frames, control_b, exit_b, recipe, reference
    )

    np.testing.assert_array_equal(delta_a.to_numpy(), delta_b.to_numpy())
    np.testing.assert_array_equal(q1_a.to_numpy(), q1_b.to_numpy())
    for group in TARGET_COLS:
        key = "new_cf_control_vs_canonical_kwh_q07_report_only"
        assert key in audit_a[group]
        assert audit_a[group][key] != audit_b[group][key]
        assert audit_a[group][key]["used_by_gate_abort_formula_or_selection"] is False


def test_seven_veto_gates_are_strict_and_have_no_extra_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2024-01-01 01:00", periods=7, freq="h")
    base = pd.DataFrame(0.0, index=index, columns=TARGET_COLS)
    candidate = base.copy()
    labels = base.copy()
    deltas = {
        name: {"score": 0.001, "one_minus_nmae": 0.0, "ficr": 0.001}
        for name in ("FULL", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
    }
    order = list(deltas)
    calls = 0

    def fake_metric(frame: pd.DataFrame, _labels: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
        nonlocal calls
        name = order[calls // 2]
        is_candidate = calls % 2 == 1
        calls += 1
        value = deltas[name] if is_candidate else {"score": 0.0, "one_minus_nmae": 0.0, "ficr": 0.0}
        return {**value, "rows": int(np.count_nonzero(mask)), "by_group": {}}

    monkeypatch.setattr(runner, "_metric", fake_metric)
    result = runner._score_veto(base, candidate, labels)
    assert tuple(result["slices"]) == tuple(order)
    assert result["gates"] == {
        "all_seven_score_positive": True,
        "full_n_nonnegative": True,
        "full_f_positive": True,
    }
    assert result["passed"] is True

    for field in ("score", "ficr"):
        calls = 0
        deltas["Q4" if field == "score" else "FULL"][field] = 0.0
        result = runner._score_veto(base, candidate, labels)
        assert result["passed"] is False
        deltas["Q4" if field == "score" else "FULL"][field] = 0.001
    calls = 0
    deltas["FULL"]["one_minus_nmae"] = -np.finfo(float).eps
    assert runner._score_veto(base, candidate, labels)["passed"] is False


def test_operating_quarter_masks_partition_boundary_hours_exactly() -> None:
    index = runner.STRESS_INDEX
    labels = pd.DataFrame(
        {group: np.full(len(index), 0.50 * CAPACITY_KWH[group]) for group in TARGET_COLS},
        index=index,
    )
    base = labels.copy()
    result = runner._score_veto(base, base.copy(), labels)
    expected_rows = {
        "FULL": 8784,
        "H1": 4368,
        "H2": 4416,
        "Q1": 2184,
        "Q2": 2184,
        "Q3": 2208,
        "Q4": 2208,
    }
    assert {name: row["base"]["rows"] for name, row in result["slices"].items()} == expected_rows


class _TrackingStream(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.operations: list[tuple[str, int, int]] = []

    def read(self, size: int = -1) -> bytes:
        start = self.tell()
        value = super().read(size)
        self.operations.append(("read", start, len(value)))
        return value

    def seek(self, offset: int, whence: int = 0) -> int:
        result = super().seek(offset, whence)
        self.operations.append(("seek", result, 0))
        return result


class _Openable:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.streams: list[_TrackingStream] = []

    def open(self, *_args: Any, **_kwargs: Any) -> _TrackingStream:
        stream = _TrackingStream(self.payload)
        self.streams.append(stream)
        return stream


def _prefix_csv_bytes() -> bytes:
    index = pd.date_range("2022-01-01 01:00", "2024-01-01 00:00", freq="h")
    frame = pd.DataFrame({"kst_dtm": index.strftime("%Y-%m-%d %H:%M:%S")})
    for group in TARGET_COLS:
        frame[group] = 1.0
    return frame.to_csv(index=False, lineterminator="\n").encode()


def _suffix_csv_bytes() -> bytes:
    frame = pd.DataFrame({"kst_dtm": runner.STRESS_INDEX.strftime("%Y-%m-%d %H:%M:%S")})
    for group in TARGET_COLS:
        frame[group] = 1.0
    return frame.to_csv(index=False, header=False, lineterminator="\n").encode()


def test_prefix_reader_never_touches_sentinel_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = _prefix_csv_bytes()
    source = _Openable(prefix + b"DO_NOT_TOUCH_SUFFIX")
    monkeypatch.setattr(runner, "RAW_LABELS", source)
    monkeypatch.setattr(runner, "PREFIX_BYTES", len(prefix))
    monkeypatch.setattr(runner, "PREFIX_SHA", hashlib.sha256(prefix).hexdigest())

    raw, audit = runner._read_prefix_raw()
    frame = runner._parse_prefix(raw)

    assert raw == prefix
    assert len(frame) == 17_520
    assert audit["suffix_bytes_read"] == 0
    assert len(source.streams) == 1
    assert source.streams[0].operations == [("read", 0, len(prefix))]


def test_suffix_reader_reads_once_parses_once_and_hashes_retained_buffers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    prefix = b"LOCKED_PREFIX"
    suffix = _suffix_csv_bytes()
    source = tmp_path / "bounded_labels.csv"
    source.write_bytes(prefix + suffix)
    monkeypatch.setattr(runner, "RAW_LABELS", source)
    monkeypatch.setattr(runner, "PREFIX_BYTES", len(prefix))
    monkeypatch.setattr(runner, "SUFFIX_BYTES", len(suffix))
    monkeypatch.setattr(runner, "FULL_LABEL_BYTES", len(prefix) + len(suffix))
    monkeypatch.setattr(runner, "SUFFIX_SHA", hashlib.sha256(suffix).hexdigest())
    monkeypatch.setattr(runner, "FULL_LABEL_SHA", hashlib.sha256(prefix + suffix).hexdigest())
    original_read_csv = runner.pd.read_csv
    parse_count = 0

    def counted_read_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        nonlocal parse_count
        parse_count += 1
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(runner.pd, "read_csv", counted_read_csv)
    frame, raw, audit = runner._read_suffix_once(prefix)

    assert raw == suffix
    assert frame.index.equals(runner.STRESS_INDEX)
    assert parse_count == 1
    assert audit["disk_reads"] == 1
    assert audit["physical_file_size_bytes"] == len(prefix) + len(suffix)
    assert audit["full_file_size_exact"] is True


def test_suffix_reader_rejects_appended_tail_without_reading_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prefix = b"LOCKED_PREFIX"
    suffix = _suffix_csv_bytes()
    source = tmp_path / "labels_with_appended_tail.csv"
    source.write_bytes(prefix + suffix + b"APPENDED_TAIL_MUST_NOT_BE_ACCEPTED")
    monkeypatch.setattr(runner, "RAW_LABELS", source)
    monkeypatch.setattr(runner, "PREFIX_BYTES", len(prefix))
    monkeypatch.setattr(runner, "SUFFIX_BYTES", len(suffix))
    monkeypatch.setattr(runner, "FULL_LABEL_BYTES", len(prefix) + len(suffix))
    monkeypatch.setattr(runner, "SUFFIX_SHA", hashlib.sha256(suffix).hexdigest())
    monkeypatch.setattr(runner, "FULL_LABEL_SHA", hashlib.sha256(prefix + suffix).hexdigest())

    with pytest.raises(AssertionError, match="suffix differs"):
        runner._read_suffix_once(prefix)


def test_fit_pair_uses_one_mask_and_two_independent_reload_predictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = pd.date_range("2023-01-01", periods=4, freq="h")
    apply_index = pd.date_range("2024-01-01", periods=2, freq="h")
    features = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}, index=index, dtype=np.float32)
    apply = pd.DataFrame({"a": [2, 4], "b": [1, 3]}, index=apply_index, dtype=np.float32)
    labels = pd.Series([3000.0, np.nan, 4000.0, 5000.0], index=index)
    fitted: list[tuple[pd.DatetimeIndex, np.ndarray, dict[str, Any]]] = []
    saved: dict[Path, Any] = {}
    loads: list[Path] = []

    class FakeModel:
        def __init__(self, **params: Any) -> None:
            self.params = params

        def fit(self, x: pd.DataFrame, y: np.ndarray) -> "FakeModel":
            fitted.append((x.index.copy(), np.asarray(y).copy(), self.params))
            self.width = x.shape[1]
            return self

        def predict(self, x: pd.DataFrame) -> np.ndarray:
            return x.iloc[:, 0].to_numpy(float) + self.width

    def save(path: Path, value: Any) -> None:
        saved[path] = value

    def load(path: Path) -> Any:
        loads.append(path)
        return saved[path]

    monkeypatch.setattr(runner, "LGBMRegressor", FakeModel)
    monkeypatch.setattr(runner, "_atomic_joblib", save)
    monkeypatch.setattr(runner.joblib, "load", load)
    monkeypatch.setattr(runner, "describe_file", lambda path: {"path": str(path), "size_bytes": 1, "sha256": "x"})

    control, exited, records = runner._fit_pair(
        "kpx_group_1", features, labels, apply, ["a", "b"], ["a"], tmp_path, 3
    )

    assert len(fitted) == 2
    assert fitted[0][0].equals(fitted[1][0])
    np.testing.assert_array_equal(fitted[0][1], fitted[1][1])
    assert fitted[0][2] == fitted[1][2] == runner.PARAMS
    assert len(loads) == 4
    assert records["control"]["eligible_index_target_digest"] == records["nonwind"]["eligible_index_target_digest"]
    np.testing.assert_array_equal(control, apply["a"].to_numpy(float) + 2)
    np.testing.assert_array_equal(exited, apply["a"].to_numpy(float) + 1)


def test_fit_pair_rejects_any_non_bit_exact_reload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    index = pd.date_range("2023-01-01", periods=2, freq="h")
    features = pd.DataFrame({"a": [1.0, 2.0]}, index=index)
    labels = pd.Series([3000.0, 4000.0], index=index)
    saved: dict[Path, Any] = {}
    load_number = 0

    class Model:
        def __init__(self, **_params: Any) -> None:
            pass

        def fit(self, _x: pd.DataFrame, _y: np.ndarray) -> "Model":
            return self

        def predict(self, x: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(x))

    class Changed(Model):
        def predict(self, x: pd.DataFrame) -> np.ndarray:
            return np.ones(len(x))

    def load(path: Path) -> Any:
        nonlocal load_number
        load_number += 1
        return {**saved[path], "model": Changed()} if load_number == 2 else saved[path]

    monkeypatch.setattr(runner, "LGBMRegressor", Model)
    monkeypatch.setattr(runner, "_atomic_joblib", lambda path, value: saved.__setitem__(path, value))
    monkeypatch.setattr(runner.joblib, "load", load)
    with pytest.raises(AssertionError, match="save/reload"):
        runner._fit_pair("kpx_group_1", features, labels, features, ["a"], ["a"], tmp_path, 2)


def test_canonical_q07_replay_is_bit_exact_and_mismatch_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2025-01-01 01:00", periods=2, freq="h")
    features = {group: pd.DataFrame({"x": [1.0, 2.0]}, index=index) for group in TARGET_COLS}

    class Model:
        def predict(self, x: pd.DataFrame) -> np.ndarray:
            return x["x"].to_numpy(float) * 100.0

    payload = {
        "feature_names": {group: ["x"] for group in TARGET_COLS},
        "models": {group: Model() for group in TARGET_COLS},
    }
    stored = pd.DataFrame({group: [100.0, 200.0] for group in TARGET_COLS}, index=index)
    monkeypatch.setattr(runner, "_identity", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner.joblib, "load", lambda _path: payload)
    result = runner._canonical_q07_replay(features, stored)
    assert all(row == {"array_equal": True, "max_abs_kwh": 0.0} for row in result.values())
    changed = stored.copy()
    changed.iloc[0, 0] = np.nextafter(100.0, np.inf)
    with pytest.raises(AssertionError, match="canonical q07 replay differs"):
        runner._canonical_q07_replay(features, changed)


def test_csv_is_bom_lf_six_decimal_roundtrip_and_no_overwrite(tmp_path: Path) -> None:
    index = runner.TEST_INDEX
    sample = pd.DataFrame(
        {
            "forecast_id": pd.Series([f"{value:06d}" for value in range(len(index))], dtype="string"),
            "forecast_kst_dtm": pd.Series(index.strftime("%Y-%m-%d %H:%M:%S"), dtype="string"),
            **{group: np.zeros(len(index)) for group in TARGET_COLS},
        }
    )
    prediction = pd.DataFrame(
        {group: np.linspace(0.1234564, 100.9876544, len(index)) for group in TARGET_COLS},
        index=index,
    )
    path = tmp_path / "submission.csv"

    audit = runner._write_csv(path, sample, prediction)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert audit["text_roundtrip_exact"] is True
    first_data = raw.decode("utf-8-sig").splitlines()[1].split(",")
    assert first_data[0] == "000000"
    assert all(re.fullmatch(r"-?\d+\.\d{6}", token) for token in first_data[-3:])
    with pytest.raises(FileExistsError):
        runner._write_csv(path, sample, prediction)


def test_csv_rejects_nonfinite_values(tmp_path: Path) -> None:
    index = runner.TEST_INDEX
    sample = pd.DataFrame(
        {
            "forecast_id": [str(i) for i in range(len(index))],
            "forecast_kst_dtm": index.astype(str),
            **{group: np.zeros(len(index)) for group in TARGET_COLS},
        }
    )
    prediction = pd.DataFrame(0.0, index=index, columns=TARGET_COLS)
    prediction.iloc[0, 0] = np.nan
    with pytest.raises((AssertionError, ValueError), match="finite"):
        runner._write_csv(tmp_path / "bad.csv", sample, prediction)


def test_static_audit_has_no_data_access_and_cli_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    science = ({"delta_transfer_formula": {}}, {"correction": {"incorrect_v1_stress_reference": {"must_not_be_used_by_this_experiment": True}}}, {"effective_label_access_order": []})
    monkeypatch.setattr(runner, "_verify_science", lambda: science)
    monkeypatch.setattr(runner, "ATTEMPT", tmp_path / "attempt")
    monkeypatch.setattr(runner, "HEAVY_GUARD", tmp_path / "guard")
    result = runner.static_audit(tmp_path / "out")
    assert result["verdict"] == "PASS"
    assert result["science_or_data_access"] == 0
    for argv in ([], ["--run", "--static-audit"], ["--unknown"]):
        with pytest.raises(SystemExit):
            runner.parse_args(argv)


def test_real_run_requires_execution_amendment_and_canonical_output_namespace(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--run"])
    with pytest.raises(AssertionError, match="canonical frozen output"):
        runner.main(
            [
                "--run",
                "--execution-amendment",
                str(runner.EXECUTION_AMENDMENT),
                "--out-dir",
                str(tmp_path / "alternate"),
            ]
        )


def test_execution_authorization_verifier_accepts_only_hash_bound_frozen_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    amendment = tmp_path / "execution_v4.json"
    review = tmp_path / "review.json"
    out = tmp_path / "canonical-output"
    monkeypatch.setattr(runner, "EXECUTION_AMENDMENT", amendment)
    monkeypatch.setattr(runner, "PRELAUNCH_REVIEW", review)
    monkeypatch.setattr(runner, "ATTEMPT", tmp_path / "attempt")
    monkeypatch.setattr(runner, "HEAVY_GUARD", tmp_path / "guard")
    monkeypatch.setattr(runner, "_authorization_data_declarations", lambda: [])
    monkeypatch.setattr(
        runner,
        "_identity",
        lambda spec, **_kwargs: {
            "path": str(Path(spec[0]).resolve()),
            "size_bytes": int(spec[1]),
            "sha256": str(spec[2]),
            "hash_verified": True,
        },
    )
    roles = {
        "runner", "independent_postrun_auditor", "focused_runner_tests",
        "independent_auditor_tests", "root_contract_tests", "src_metric",
        "src_virtual_feature_exit", "src_density_ratio", "src_manifest",
        "src_features", "build_features", "info_xlsx",
    }
    canonical_paths = {
        role: (Path(runner.__file__).resolve() if role == "runner" else (tmp_path / f"{role}.txt").resolve())
        for role in roles
    }
    monkeypatch.setattr(runner, "_authorization_bound_role_paths", lambda: canonical_paths)
    bound_files = []
    for role in sorted(roles):
        path = canonical_paths[role]
        bound_files.append({"role": role, "path": str(path), "size_bytes": 1, "sha256": f"sha-{role}"})
    payload = {
        "schema_version": 1,
        "experiment_id": runner.EXPERIMENT_ID,
        "status": "AUTHORIZED_SINGLE_EXECUTION",
        "single_attempt": True,
        "canonical_output": str(out.resolve()),
        "science_heads": [
            runner._declared_identity(f"science_v{i}", spec)
            for i, spec in enumerate(runner.SCIENCE_FILES, start=1)
        ],
        "data_declarations": [],
        "bound_files": bound_files,
        "runtime": {
            "python": runner.platform.python_version(),
            "packages": runner.package_versions(
                ("numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow", "joblib")
            ),
        },
        "prelaunch_zero_state": {
            "canonical_output_absent": True,
            "attempt_absent": True,
            "heavy_guard_absent": True,
        },
        "required_independent_prelaunch_review": str(review.resolve()),
    }
    runner_record = next(row for row in bound_files if row["role"] == "runner")

    def freeze(candidate: dict[str, Any]) -> str:
        amendment.write_text(json.dumps(candidate, sort_keys=True) + "\n", encoding="utf-8")
        amendment_sha = hashlib.sha256(amendment.read_bytes()).hexdigest()
        amendment.with_suffix(amendment.suffix + ".sha256").write_text(
            f"{amendment_sha}  {amendment.name}\n", encoding="utf-8"
        )
        review.write_text(
            json.dumps(
                {
                    "experiment_id": runner.EXPERIMENT_ID,
                    "verdict": "PASS",
                    "blocking_defects": 0,
                    "execution_amendment": {"sha256": amendment_sha},
                    "bound_runner": {"sha256": runner_record["sha256"]},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return amendment_sha

    amendment_sha = freeze(payload)

    result = runner._verify_execution_authorization(amendment, out)
    assert result["file"]["sha256"] == amendment_sha
    assert {row["role"] for row in result["verified_bound_files"]} == roles

    duplicate = json.loads(json.dumps(payload))
    duplicate["bound_files"].append(dict(duplicate["bound_files"][0]))
    freeze(duplicate)
    with pytest.raises(AssertionError, match="bound-file roles"):
        runner._verify_execution_authorization(amendment, out)

    masquerade = json.loads(json.dumps(payload))
    row = next(item for item in masquerade["bound_files"] if item["role"] == "src_metric")
    row["path"] = str((tmp_path / "self-declared-arbitrary.py").resolve())
    freeze(masquerade)
    with pytest.raises(AssertionError, match="canonical path"):
        runner._verify_execution_authorization(amendment, out)

    relative_output = json.loads(json.dumps(payload))
    relative_output["canonical_output"] = os.path.relpath(out, Path.cwd())
    assert Path(relative_output["canonical_output"]).resolve() == out.resolve()
    freeze(relative_output)
    with pytest.raises(AssertionError, match="output differs"):
        runner._verify_execution_authorization(amendment, out)

    freeze(payload)
    amendment.with_suffix(amendment.suffix + ".sha256").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="sidecar"):
        runner._verify_execution_authorization(amendment, out)


def test_source_and_stress_locks_precede_protected_reads_and_suffix_is_single_use() -> None:
    source = inspect.getsource(runner.run)
    source_lock = source.index('"source_lock.json"')
    prefix_raw = source.index("_read_prefix_raw(")
    prefix_parse = source.index("_parse_prefix(")
    train_array = source.index("_read_cache(TRAIN_CACHES")
    candidate_lock = source.index('"stress/candidate_lock.json"')
    suffix_read = source.index("_read_suffix_once(")
    final_test_read = source.index("_read_cache(TEST_CACHES")
    assert prefix_raw < source_lock < prefix_parse
    assert source_lock < train_array
    assert candidate_lock < suffix_read
    assert source.count("_read_suffix_once(") == 1
    assert '"stress/manifest.json"' in source
    assert source.index('"stress/manifest.json"') < final_test_read
    assert "pd.concat([prefix_labels, suffix_labels])" in source
    pre_gate = source[: source.index("if not stress_result")]
    assert '_declared_identity(f"official_test_raw_{i}", spec)' in pre_gate
    assert "_identity(spec, hash_file=False)" not in pre_gate


def test_original_gate_a0_is_reconstructed_and_locked_before_first_stress_fit() -> None:
    source = inspect.getsource(runner.run)
    fit = source.index("_fit_pair(")
    markers = (
        "prefit_A0_reconstruction_verified",
        "prefit_a0_reconstruction_verified",
        "_verify_A0_reconstruction",
        "_verify_prefit_a0",
        "_verify_a0_before_fit",
    )
    positions = [source.index(marker) for marker in markers if marker in source]
    assert positions, "runner has no explicit pre-fit original-gate A0 reconstruction lock"
    assert min(positions) < fit


def test_failed_veto_executes_no_final_reader_fit_or_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "attempt"
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(_recipe()), encoding="utf-8")
    prefix_index = pd.date_range("2023-12-31 23:00", periods=2, freq="h", name="forecast_kst_dtm")
    prefix_labels = pd.DataFrame(3000.0, index=prefix_index, columns=TARGET_COLS)
    train_frame = pd.DataFrame({"x": np.zeros(len(runner.STRESS_INDEX), dtype=np.float32)}, index=runner.STRESS_INDEX)
    zero = pd.DataFrame(0.0, index=runner.STRESS_INDEX, columns=TARGET_COLS)
    counters = {"cache": 0, "prediction": 0, "fit_pair": 0, "suffix": 0, "csv": 0, "canonical": 0}
    suffix_opened = False

    class Guard:
        def __enter__(self) -> "Guard":
            return self

        def verify(self) -> None:
            return None

        def __exit__(self, *_args: Any) -> None:
            return None

    v1 = {"delta_transfer_formula": {"formula": "locked"}}
    monkeypatch.setattr(runner, "_verify_science", lambda: (v1, {}, {}))
    def create_attempt(path: Path) -> dict[str, Any]:
        path.mkdir(parents=True)
        return {"path": str(tmp_path / "attempt-lock"), "size_bytes": 0, "sha256": "mock"}

    monkeypatch.setattr(runner, "_create_attempt", create_attempt)
    monkeypatch.setattr(runner, "HeavyGuard", Guard)
    monkeypatch.setattr(runner, "_copy_science", lambda _out: [])
    monkeypatch.setattr(runner, "_copy_execution_lineage", lambda _out, _authorization: {"authorization": "mock"})
    monkeypatch.setattr(
        runner,
        "_identity",
        lambda spec, **_kwargs: {"path": str(spec[0]), "size_bytes": spec[1], "sha256": spec[2]},
    )
    monkeypatch.setattr(runner, "_read_prefix_raw", lambda: (b"prefix", {"suffix_bytes_read": 0}))
    monkeypatch.setattr(runner, "_parse_prefix", lambda _raw: prefix_labels)
    monkeypatch.setattr(runner, "GATE_RECIPE", (recipe_path, 0, "x"))
    monkeypatch.setattr(runner, "FINAL_RECIPE", (recipe_path, 0, "x"))
    monkeypatch.setattr(runner, "describe_file", lambda path: {"path": str(path), "size_bytes": 0, "sha256": "mock"})

    def read_cache(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        if suffix_opened:
            raise AssertionError("final/test cache reader opened after failed veto")
        counters["cache"] += 1
        return train_frame

    def read_prediction(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        if suffix_opened:
            raise AssertionError("final component reader opened after failed veto")
        counters["prediction"] += 1
        return zero.copy()

    def fit_pair(*_args: Any, **_kwargs: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if suffix_opened:
            raise AssertionError("final model fit opened after failed veto")
        counters["fit_pair"] += 1
        values = np.zeros(len(runner.STRESS_INDEX))
        return values, values.copy(), {
            "control": {"file": {"path": "control", "size_bytes": 0, "sha256": "mock"}},
            "nonwind": {"file": {"path": "nonwind", "size_bytes": 0, "sha256": "mock"}},
        }

    def read_suffix(_prefix: bytes) -> tuple[pd.DataFrame, bytes, dict[str, Any]]:
        nonlocal suffix_opened
        suffix_opened = True
        counters["suffix"] += 1
        return zero.copy(), b"suffix", {"disk_reads": 1}

    def forbidden_csv(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        counters["csv"] += 1
        raise AssertionError("CSV writer reached after failed veto")

    def forbidden_canonical(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        counters["canonical"] += 1
        raise AssertionError("canonical final model opened after failed veto")

    monkeypatch.setattr(runner, "_read_cache", read_cache)
    monkeypatch.setattr(runner, "_feature_contract", lambda _frame: (["x"], ["x"]))
    monkeypatch.setattr(runner, "_read_prediction", read_prediction)
    monkeypatch.setattr(runner, "_verify_A0_reconstruction", lambda *_args: {group: {} for group in TARGET_COLS})
    monkeypatch.setattr(runner, "_fit_pair", fit_pair)
    monkeypatch.setattr(
        runner,
        "_assemble_delta",
        lambda *_args: (zero.copy(), zero.copy(), zero.copy(), zero.copy(), {group: {} for group in TARGET_COLS}),
    )
    monkeypatch.setattr(runner, "_atomic_parquet", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_verify_candidate_durability", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_read_suffix_once", read_suffix)
    monkeypatch.setattr(
        runner,
        "_score_veto",
        lambda *_args: {"passed": False, "gates": {"all_seven_score_positive": False}},
    )
    monkeypatch.setattr(runner, "_write_csv", forbidden_csv)
    monkeypatch.setattr(runner, "_canonical_q07_replay", forbidden_canonical)

    assert runner.run(out, {"authorization": "synthetic"}) == 2
    assert counters == {
        "cache": 3,
        "prediction": 8,
        "fit_pair": 3,
        "suffix": 1,
        "csv": 0,
        "canonical": 0,
    }
    for relative in (
        "stress/rejection.json",
        "stress/manifest.json",
        "access_ledger.json",
        "manifest.json",
    ):
        assert (out / relative).is_file()
    access = json.loads((out / "access_ledger.json").read_text(encoding="utf-8"))
    assert access["final_model_fits"] == 0
    assert access["test_cache_files_opened"] == 0
    assert access["final_component_files_opened"] == 0
    assert access["sample_files_opened"] == 0
    assert access["csv_files_written"] == 0

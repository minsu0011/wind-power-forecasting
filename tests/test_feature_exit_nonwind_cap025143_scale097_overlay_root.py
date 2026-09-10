from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_feature_exit_nonwind_cap025143_scale097_overlay.py"
SPEC = importlib.util.spec_from_file_location("feature_exit_nonwind_overlay_runner_root", PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_static_audit_is_zero_access(tmp_path: Path) -> None:
    result = mod.static_audit(tmp_path / "never-created")
    assert result["verdict"] == "PASS"
    assert result["science_or_data_access"] == 0


def _simple_recipe() -> dict:
    recipe = {
        "ensemble": {
            "weights": {},
            "affine": {},
            "clip": {},
            "power_bins": {},
        }
    }
    for group in mod.TARGET_COLS:
        recipe["ensemble"]["weights"][group] = {
            "lgb_l1": 0.0,
            "lgb_q07": 1.0,
            "shared_l1": 0.0,
            "shared_q07": 0.0,
            "top200_q07": 0.0,
            "energy_q06": 0.0,
        }
        recipe["ensemble"]["affine"][group] = {"scale": 1.0, "bias_kwh": 0.0}
        recipe["ensemble"]["clip"][group] = {"lower_capacity_fraction": 0.0, "upper_capacity_fraction": 1.02}
        recipe["ensemble"]["power_bins"][group] = None
    recipe["ensemble"]["power_bins"]["kpx_group_2"] = {
        "edges_cf": [0.0, 0.25, 0.50, 0.75, 1.02],
        "delta_kwh": [-400.0, -1000.0, 1200.0, -200.0],
    }
    return recipe


def test_cap_outer_clip_and_g2_baseline_bin_freeze() -> None:
    index = pd.date_range("2025-01-01 01:00", periods=4, freq="h", name="forecast_kst_dtm")
    recipe = _simple_recipe()
    component_frames = {}
    for name in recipe["ensemble"]["weights"]["kpx_group_1"]:
        component_frames[name] = pd.DataFrame(0.0, index=index, columns=mod.TARGET_COLS)
    for group in mod.TARGET_COLS:
        cap = mod.CAPACITY_KWH[group]
        component_frames["lgb_q07"].loc[:, group] = np.array([0.249, 0.499, 0.749, 1.019]) * cap
    control = pd.DataFrame(0.0, index=index, columns=mod.TARGET_COLS)
    exit_ = pd.DataFrame(1.0, index=index, columns=mod.TARGET_COLS)
    reference = pd.DataFrame(index=index, columns=mod.TARGET_COLS, dtype=float)
    for group in mod.TARGET_COLS:
        values = {name: frame[group].to_numpy(float) for name, frame in component_frames.items()}
        reference[group] = mod.assemble_locked_group(values, group=group, capacity_kwh=mod.CAPACITY_KWH[group], ensemble=recipe["ensemble"])
    delta, q1, a0, a1, audit = mod._assemble_delta(component_frames, control, exit_, recipe, reference)
    for group in mod.TARGET_COLS:
        assert np.max(q1[group] / mod.CAPACITY_KWH[group]) <= 1.02
        assert audit[group]["max_abs_capped_delta_cf"] == pytest.approx(mod.CAP_CF)
    assert audit["kpx_group_2"]["hypothetical_g2_bin_crossings"] > 0
    assert np.array_equal(a1.to_numpy(float) - a0.to_numpy(float), delta.to_numpy(float))
    cap = mod.CAPACITY_KWH["kpx_group_2"]
    z0 = component_frames["lgb_q07"]["kpx_group_2"].to_numpy(float)
    z1 = q1["kpx_group_2"].to_numpy(float)
    offsets = np.array([-400.0, -1000.0, 1200.0, -200.0])
    b0 = np.searchsorted(np.array([0.25, 0.5, 0.75]), z0 / cap, side="right")
    expected = np.clip(z1 + offsets[b0], 0.0, 1.02 * cap) - np.clip(z0 + offsets[b0], 0.0, 1.02 * cap)
    assert np.array_equal(delta["kpx_group_2"].to_numpy(float), expected)


def test_g2_edges_use_side_right() -> None:
    edges = np.array([0.25, 0.50, 0.75])
    got = np.searchsorted(edges, np.array([0.25, 0.50, 0.75]), side="right")
    assert np.array_equal(got, np.array([1, 2, 3]))


def test_seven_slice_veto_passes_only_when_all_improve() -> None:
    idx = mod.STRESS_INDEX
    labels = pd.DataFrame(index=idx, columns=mod.TARGET_COLS, dtype=float)
    base = labels.copy()
    candidate = labels.copy()
    for group in mod.TARGET_COLS:
        labels[group] = 0.5 * mod.CAPACITY_KWH[group]
        base[group] = 0.2 * mod.CAPACITY_KWH[group]
        candidate[group] = labels[group]
    result = mod._score_veto(base, candidate, labels)
    assert result["passed"]
    assert set(result["slices"]) == {"FULL", "H1", "H2", "Q1", "Q2", "Q3", "Q4"}
    assert all(row["delta"]["score"] > 0 for row in result["slices"].values())
    equal = mod._score_veto(base, base, labels)
    assert not equal["passed"]


def test_veto_boundaries_assign_closing_hour_to_prior_slice() -> None:
    masks = mod._veto_masks(mod.STRESS_INDEX)
    positions = {timestamp: mod.STRESS_INDEX.get_loc(pd.Timestamp(timestamp)) for timestamp in (
        "2024-04-01 00:00:00", "2024-04-01 01:00:00",
        "2024-07-01 00:00:00", "2024-07-01 01:00:00",
        "2024-10-01 00:00:00", "2024-10-01 01:00:00",
        "2025-01-01 00:00:00",
    )}
    assert masks["Q1"][positions["2024-04-01 00:00:00"]]
    assert masks["Q2"][positions["2024-04-01 01:00:00"]]
    assert masks["H1"][positions["2024-07-01 00:00:00"]]
    assert masks["Q2"][positions["2024-07-01 00:00:00"]]
    assert masks["H2"][positions["2024-07-01 01:00:00"]]
    assert masks["Q3"][positions["2024-10-01 00:00:00"]]
    assert masks["Q4"][positions["2024-10-01 01:00:00"]]
    assert masks["H2"][positions["2025-01-01 00:00:00"]]
    assert masks["Q4"][positions["2025-01-01 00:00:00"]]
    assert all(int(masks[name].sum()) > 0 for name in ("FULL", "H1", "H2", "Q1", "Q2", "Q3", "Q4"))


def test_bounded_prefix_and_single_suffix_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prefix_index = pd.date_range("2022-01-01 01:00", "2024-01-01 00:00", freq="h")
    prefix = pd.DataFrame({"kst_dtm": prefix_index.strftime("%Y-%m-%d %H:%M:%S")})
    for group in mod.TARGET_COLS:
        prefix[group] = 0.0
    prefix_raw = prefix.to_csv(index=False, lineterminator="\n").encode()
    suffix = pd.DataFrame({"kst_dtm": mod.STRESS_INDEX.strftime("%Y-%m-%d %H:%M:%S")})
    for group in mod.TARGET_COLS:
        suffix[group] = 0.0
    suffix_raw = suffix.to_csv(index=False, header=False, lineterminator="\n").encode()
    path = tmp_path / "labels.csv"
    path.write_bytes(prefix_raw + suffix_raw)
    monkeypatch.setattr(mod, "RAW_LABELS", path)
    monkeypatch.setattr(mod, "PREFIX_BYTES", len(prefix_raw))
    monkeypatch.setattr(mod, "PREFIX_SHA", hashlib.sha256(prefix_raw).hexdigest())
    monkeypatch.setattr(mod, "SUFFIX_BYTES", len(suffix_raw))
    monkeypatch.setattr(mod, "SUFFIX_SHA", hashlib.sha256(suffix_raw).hexdigest())
    monkeypatch.setattr(mod, "FULL_LABEL_BYTES", len(prefix_raw) + len(suffix_raw))
    monkeypatch.setattr(mod, "FULL_LABEL_SHA", hashlib.sha256(prefix_raw + suffix_raw).hexdigest())
    praw, paudit = mod._read_prefix_raw()
    pframe = mod._parse_prefix(praw)
    assert len(pframe) == 17520 and paudit["suffix_bytes_read"] == 0
    sframe, sraw, saudit = mod._read_suffix_once(praw)
    assert len(sframe) == 8784 and saudit["disk_reads"] == 1
    assert praw == prefix_raw and sraw == suffix_raw


def test_csv_byte_contract(tmp_path: Path) -> None:
    idx = mod.TEST_INDEX
    sample = pd.DataFrame({
        "forecast_id": [f"forecast_{i:04d}" for i in range(1, len(idx) + 1)],
        "forecast_kst_dtm": idx.strftime("%Y-%m-%d %H:%M:%S"),
        **{group: np.zeros(len(idx)) for group in mod.TARGET_COLS},
    })
    prediction = pd.DataFrame(index=idx, columns=mod.TARGET_COLS, data=1.23456789)
    audit = mod._write_csv(tmp_path / "candidate.csv", sample, prediction)
    raw = (tmp_path / "candidate.csv").read_bytes()
    assert audit["text_roundtrip_exact"] and raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert b"1.234568" in raw


def test_cli_requires_exactly_one_mode() -> None:
    with pytest.raises(SystemExit):
        mod.parse_args([])
    with pytest.raises(SystemExit):
        mod.parse_args(["--run", "--static-audit"])

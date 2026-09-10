from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts import run_raw_attention_sficr_fixed_average_g2_rescue as runner
from src.metric import CAPACITY_KWH, TARGET_COLS


class RawAttentionSmoothFICRG2RescueTests(unittest.TestCase):
    def test_v2_preregister_and_unexecuted_v1_incident_are_exact(self) -> None:
        config, v1, _ = runner.verify_config(runner.CONFIG_PATH)
        self.assertEqual(config["supersession"]["v1_execution_count"], 0)
        self.assertEqual(config["supersession"]["scientific_retune_count"], 0)
        self.assertTrue(v1["risk_classification"]["selection_unsafe"])
        incident = json.loads(runner.INCIDENT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(incident["ledger_at_supersession"]["models_fitted"], 0)

    def test_upstream_training_schedule_is_immutable(self) -> None:
        self.assertEqual(tuple(runner.RAW_SEEDS), (42, 2026))
        self.assertEqual(tuple(runner.SF_SEEDS), (42, 2026))
        self.assertEqual((runner.RAW_EPOCHS, runner.SF_EPOCHS), (240, 240))
        self.assertEqual((runner.RAW_BATCH_SIZE, runner.SF_BATCH_SIZE), (32, 32))

    @staticmethod
    def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        index = runner.YEAR_2024
        count = len(index)
        primary = pd.DataFrame(
            {
                TARGET_COLS[0]: np.linspace(100.0, 200.0, count),
                TARGET_COLS[1]: np.linspace(5000.0, 7000.0, count),
                TARGET_COLS[2]: np.linspace(300.0, 600.0, count),
            }, index=index,
        )
        recent = primary + pd.DataFrame(
            {TARGET_COLS[0]: 3.0, TARGET_COLS[1]: 7.0, TARGET_COLS[2]: 11.0}, index=index
        )
        raw = pd.DataFrame(0.4, index=index, columns=TARGET_COLS)
        smooth = pd.DataFrame(0.5, index=index, columns=TARGET_COLS)
        return primary, recent, raw, smooth

    def test_fixed_formula_and_transfer_are_exact(self) -> None:
        primary, recent, raw, smooth = self._frames()
        p, r, delta = runner.fixed_actions(primary, recent, raw, smooth)
        base = primary[runner.G2].to_numpy(float)
        expected = np.clip(
            base
            + 0.0125 * (0.4 * CAPACITY_KWH[runner.G2] - base)
            + 0.0125 * (0.5 * CAPACITY_KWH[runner.G2] - base),
            0.0, 1.02 * CAPACITY_KWH[runner.G2],
        )
        self.assertTrue(np.array_equal(p[runner.G2].to_numpy(), expected))
        self.assertTrue(np.array_equal(delta.to_numpy(), expected - base))
        self.assertTrue(np.array_equal(r[runner.G2].to_numpy(), recent[runner.G2] + delta))

    def test_identity_groups_are_value_bit_exact(self) -> None:
        primary, recent, raw, smooth = self._frames()
        p, r, _ = runner.fixed_actions(primary, recent, raw, smooth)
        for group in (TARGET_COLS[0], TARGET_COLS[2]):
            self.assertEqual(p[group].to_numpy().tobytes(), primary[group].to_numpy().tobytes())
            self.assertEqual(r[group].to_numpy().tobytes(), recent[group].to_numpy().tobytes())

    def test_both_actions_use_capacity_clipping(self) -> None:
        primary, recent, raw, smooth = self._frames()
        primary[runner.G2] = 1.02 * CAPACITY_KWH[runner.G2]
        recent[runner.G2] = 1.02 * CAPACITY_KWH[runner.G2]
        raw[runner.G2] = 9.0
        smooth[runner.G2] = 9.0
        p, r, _ = runner.fixed_actions(primary, recent, raw, smooth)
        self.assertTrue((p[runner.G2] <= 1.02 * CAPACITY_KWH[runner.G2]).all())
        self.assertTrue((r[runner.G2] <= 1.02 * CAPACITY_KWH[runner.G2]).all())

    def test_misaligned_candidate_input_is_rejected(self) -> None:
        primary, recent, raw, smooth = self._frames()
        with self.assertRaises(ValueError):
            runner.fixed_actions(primary.iloc[:-1], recent, raw, smooth)

    @staticmethod
    def _gate_records(delta: float = 0.01, component: float = 0.01) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for baseline in ("primary_v3", "recent_v4"):
            group: dict[str, dict] = {}
            mixed: dict[str, dict] = {}
            for name in runner.SLICES:
                group_candidate_score = 0.5 + delta
                mixed_candidate_score = 0.6 + delta
                group[name] = {
                    "baseline": {"score": 0.5, "one_minus_nmae": 0.7, "ficr": 0.3},
                    "candidate": {
                        "score": group_candidate_score,
                        "one_minus_nmae": 0.7 + (component if name == "full" else 0.0),
                        "ficr": 0.3 + (component if name == "full" else 0.0),
                    },
                    "delta": group_candidate_score - 0.5,
                }
                mixed[name] = {
                    "baseline": {"total_score": 0.6, "one_minus_nmae": 0.8, "ficr": 0.4},
                    "candidate": {
                        "total_score": mixed_candidate_score,
                        "one_minus_nmae": 0.8 + (component if name == "full" else 0.0),
                        "ficr": 0.4 + (component if name == "full" else 0.0),
                    },
                    "delta": mixed_candidate_score - 0.6,
                }
            result[baseline] = {"g2": group, "mixed": mixed}
        return result

    def test_dual_gate_requires_all_registered_conditions(self) -> None:
        observed = runner.dual_gate(self._gate_records())
        self.assertTrue(observed["dual_baseline_promoted"])
        self.assertEqual(observed["decision"], "AWAIT_INDEPENDENT_AUDIT")

    def test_dual_gate_rejects_one_slice_or_component_failure(self) -> None:
        records = self._gate_records()
        row = records["recent_v4"]["mixed"]["Q4"]
        row["candidate"]["total_score"] = row["baseline"]["total_score"]
        row["delta"] = 0.0
        self.assertFalse(runner.dual_gate(records)["dual_baseline_promoted"])
        records = self._gate_records(component=-0.01)
        self.assertFalse(runner.dual_gate(records)["dual_baseline_promoted"])

    def test_score_reader_is_fail_armed_before_global_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            absent = Path(directory) / "absent.json"
            with mock.patch.object(runner.raw_protocol, "_read_full_labels") as reader:
                with self.assertRaises(RuntimeError):
                    runner._read_score_labels_after_lock(Path(directory), {}, absent)
                reader.assert_not_called()

    def test_source_order_locks_candidate_before_score_reader(self) -> None:
        source = inspect.getsource(runner.run_stage2)
        self.assertLess(source.index("candidate_lock_before_2024_labels.json"), source.index("_read_score_labels_after_lock"))
        self.assertLess(source.index("_read_score_labels_after_lock"), source.index("_comparison("))
        self.assertIn("no_2025_or_csv", source)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.run_turbine_scada_power import (
    PREREGISTER_SHA256,
    _aggregate_turbine_hourly,
    _blend,
    _candidate_key,
    _discover_csv_prefix,
    _passing_and_selection,
    _read_scada_prefix,
)
from src.manifest import sha256_file


class TurbineSCADAPowerTests(unittest.TestCase):
    def test_preregister_hash_is_immutable(self) -> None:
        path = Path("configs/turbine_scada_power_preregister.json")
        self.assertEqual(sha256_file(path), PREREGISTER_SHA256)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["candidate_grid"]["total_candidates_per_group"], 6)

    def test_prefix_reader_stops_at_boundary_row_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scada.csv"
            content = (
                "kst_dtm,value\n"
                "2022-12-31 23:50:00,123\n"
                "2023-01-01 00:00:00,SECRET_FUTURE_VALUE\n"
            )
            path.write_text(content, encoding="utf-8", newline="")
            record = _discover_csv_prefix(path, pd.Timestamp("2023-01-01 00:00:00"))
            expected = content.encode("utf-8").index(b"2023-01-01 00:00:00")
            self.assertEqual(record["prefix_bytes"], expected)
            self.assertEqual(record["prefix_data_rows"], 1)
            self.assertEqual(record["suffix_bytes_exposed_to_parser"], 0)
            frame, evidence = _read_scada_prefix(record)
            self.assertEqual(len(frame), 1)
            self.assertEqual(evidence["underlying_file_position_after_parse"], expected)

    def test_vestas_spikes_become_missing_and_alignment_is_plus_one_hour(self) -> None:
        times = pd.date_range("2022-01-01 00:00:00", periods=6, freq="10min")
        frame = pd.DataFrame({"kst_dtm": times})
        for turbine in range(1, 7):
            values = np.full(6, 100.0)
            if turbine == 1:
                values[2] = 51_770_425.0
            frame[f"vestas_wtg{turbine:02d}_power_kw10m"] = values
        hourly, audit = _aggregate_turbine_hourly(frame, "kpx_group_1")
        self.assertEqual(hourly.index[0], pd.Timestamp("2022-01-01 01:00:00"))
        self.assertTrue(np.isnan(hourly.iloc[0, 0]))
        self.assertEqual(float(hourly.iloc[0, 1]), 600.0)
        self.assertEqual(audit["total_spikes_removed"], 1)

    def test_selection_requires_every_slice_strictly_positive(self) -> None:
        comparisons = {
            "l1__w05": {
                "full": {"delta": 0.1},
                "H1": {"delta": 0.01},
                "H2": {"delta": -1e-9},
            },
            "q07__w10": {
                "full": {"delta": 0.02},
                "H1": {"delta": 0.01},
                "H2": {"delta": 0.005},
            },
            "l1__w10": {
                "full": {"delta": 0.03},
                "H1": {"delta": 0.006},
                "H2": {"delta": 0.005},
            },
        }
        passing, selected = _passing_and_selection(comparisons)
        self.assertEqual(passing, ["l1__w10", "q07__w10"])
        self.assertEqual(selected, "l1__w10")

    def test_blend_formula_and_candidate_names_are_fixed(self) -> None:
        index = pd.date_range("2023-01-01", periods=2, freq="h")
        baseline = pd.Series([1000.0, 2000.0], index=index)
        scada = pd.Series([3000.0, 1000.0], index=index)
        actual = _blend(baseline, scada, "kpx_group_1", 0.10)
        np.testing.assert_allclose(actual, [1200.0, 1900.0])
        self.assertEqual(_candidate_key("q07", 0.20), "q07__w20")


if __name__ == "__main__":
    unittest.main()

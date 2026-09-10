from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.run_turbine_spatial_supervised import (
    FEATURE_COLUMNS,
    PREREGISTER_SHA256,
    _blend,
    _idw_weights,
    _locked_groups,
    _read_weather_full_bounded,
    _spatial_turbine_features,
)
from src.features import AVAILABLE_COL, GRID_COL, TIME_COL
from src.manifest import sha256_file


class TurbineSpatialSupervisedTests(unittest.TestCase):
    def test_preregister_hash_and_single_candidate(self) -> None:
        path = Path("configs/turbine_spatial_supervised_preregister.json")
        self.assertEqual(sha256_file(path), PREREGISTER_SHA256)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["selection_contract"]["single_promotable_candidate"],
            "q07_w20",
        )
        self.assertFalse(
            payload["selection_contract"]["candidate_or_coefficient_search"]
        )

    def test_idw_weights_are_normalized_and_coordinate_specific(self) -> None:
        weights, distances = _idw_weights(
            np.array([37.0, 37.1]),
            np.array([128.0, 128.1]),
            np.array([37.001, 37.099]),
            np.array([128.001, 128.099]),
        )
        np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-15, rtol=0)
        self.assertGreater(weights[0, 0], weights[1, 0])
        self.assertGreater(weights[1, 1], weights[0, 1])
        self.assertTrue(np.isfinite(distances).all())

    @staticmethod
    def _raw_frame(source: str) -> pd.DataFrame:
        times = pd.date_range("2023-01-01 01:00:00", periods=2, freq="h")
        grids = 16 if source == "ldaps" else 9
        records = []
        for time_number, timestamp in enumerate(times):
            for grid in range(grids):
                base = float(grid + 2 * time_number)
                record = {
                    TIME_COL: timestamp,
                    AVAILABLE_COL: timestamp - pd.Timedelta(hours=3),
                    GRID_COL: grid,
                    "latitude": 37.0 + 0.01 * (grid // 4),
                    "longitude": 128.0 + 0.01 * (grid % 4),
                }
                if source == "ldaps":
                    record.update(
                        {
                            "heightAboveGround_50_50MUmax": base + 3.0,
                            "heightAboveGround_50_50MUmin": base + 1.0,
                            "heightAboveGround_50_50MVmax": base + 2.0,
                            "heightAboveGround_50_50MVmin": base,
                        }
                    )
                else:
                    record.update(
                        {
                            "heightAboveGround_100_100u": base + 2.0,
                            "heightAboveGround_100_100v": base + 1.0,
                        }
                    )
                records.append(record)
        return pd.DataFrame(records)

    @staticmethod
    def _metadata() -> dict[str, pd.DataFrame]:
        frame = pd.DataFrame(
            {
                "turbine": [1, 2],
                "latitude": [37.001, 37.029],
                "longitude": [128.001, 128.029],
                "latitude_offset": [-0.014, 0.014],
                "longitude_offset": [-0.014, 0.014],
                "rated_kwh": [10800.0, 10800.0],
            }
        )
        return {"kpx_group_1": frame}

    def test_spatial_features_have_fixed_schema_and_turbine_variation(self) -> None:
        features, audit = _spatial_turbine_features(
            {"ldaps": self._raw_frame("ldaps"), "gfs": self._raw_frame("gfs")},
            self._metadata(),
        )
        frame = features["kpx_group_1"]
        self.assertEqual(tuple(frame.columns), FEATURE_COLUMNS)
        self.assertEqual(frame.shape, (4, len(FEATURE_COLUMNS)))
        first_time = frame.xs(pd.Timestamp("2023-01-01 01:00:00"))
        self.assertNotEqual(
            float(first_time.iloc[0]["ldaps_u50_idw"]),
            float(first_time.iloc[1]["ldaps_u50_idw"]),
        )
        self.assertTrue(np.isfinite(frame.to_numpy(dtype=float)).all())
        self.assertTrue(audit["idw"]["target_free"])

    def test_promotion_requires_every_slice_strictly_positive(self) -> None:
        comparisons = {
            "kpx_group_1": {
                "full": {"delta": 0.1},
                "H1": {"delta": 0.01},
                "H2": {"delta": 0.001},
            },
            "kpx_group_2": {
                "full": {"delta": 0.1},
                "H1": {"delta": -1e-12},
                "H2": {"delta": 0.2},
            },
        }
        self.assertEqual(_locked_groups(comparisons), ["kpx_group_1"])

    def test_fixed_twenty_percent_blend(self) -> None:
        index = pd.date_range("2023-01-01", periods=2, freq="h")
        baseline = pd.Series([1000.0, 2000.0], index=index)
        spatial = pd.Series([3000.0, 1000.0], index=index)
        actual = _blend(baseline, spatial, "kpx_group_1")
        np.testing.assert_allclose(actual, [1400.0, 1800.0])

    def test_full_weather_reader_is_physically_eof_bounded(self) -> None:
        frame = self._raw_frame("gfs").iloc[:9].copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gfs.csv"
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            observed, audit = _read_weather_full_bounded(
                path,
                source="gfs",
                expected_timestamps=1,
                expected_start=pd.Timestamp("2023-01-01 01:00:00"),
                expected_end=pd.Timestamp("2023-01-01 01:00:00"),
            )
            self.assertEqual(len(observed), 9)
            self.assertEqual(audit["suffix_bytes_exposed_to_parser"], 0)
            self.assertEqual(
                audit["physical_bytes_returned"], audit["physical_byte_limit"]
            )


if __name__ == "__main__":
    unittest.main()

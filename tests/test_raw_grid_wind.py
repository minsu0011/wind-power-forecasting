from __future__ import annotations

import unittest
from unittest import mock
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.raw_grid_wind import (
    AVAILABLE_COL,
    CANDIDATE_KEYS,
    COORD_COLS,
    EXPECTED_FEATURE_COUNT,
    GFS_CHANNELS,
    GRID_COL,
    LDAPS_CHANNELS,
    RawGridWindRegressor,
    TIME_COL,
    assert_fit_before_apply,
    blend_raw_prediction,
    build_raw_grid_wind_features,
    select_group_candidate,
)
from src.manifest import sha256_file
from scripts import run_raw_grid_wind_lgb as runner


class RawGridWindFeatureTests(unittest.TestCase):
    def _catalog(self, source: str) -> list[dict[str, float | int]]:
        count = 16 if source == "ldaps" else 9
        return [
            {
                "grid_id": grid_id,
                "latitude": 37.0 + 0.01 * grid_id,
                "longitude": 128.0 + 0.02 * grid_id,
            }
            for grid_id in range(1, count + 1)
        ]

    def _raw(self, source: str, periods: int = 3) -> pd.DataFrame:
        catalog = self._catalog(source)
        channels = LDAPS_CHANNELS if source == "ldaps" else GFS_CHANNELS
        times = pd.date_range("2023-01-01 01:00", periods=periods, freq="h")
        rows: list[dict[str, object]] = []
        for time_position, time in enumerate(times):
            for node in catalog:
                row: dict[str, object] = {
                    TIME_COL: time,
                    AVAILABLE_COL: pd.Timestamp("2022-12-31 13:00"),
                    GRID_COL: node["grid_id"],
                    COORD_COLS[0]: node["latitude"],
                    COORD_COLS[1]: node["longitude"],
                }
                for channel_position, channel in enumerate(channels):
                    row[channel] = (
                        1000.0 * time_position
                        + 10.0 * int(node["grid_id"])
                        + channel_position
                    )
                rows.append(row)
        return pd.DataFrame(rows)

    def test_all_node_identities_are_distinct_and_ordered(self) -> None:
        catalog = {"ldaps": self._catalog("ldaps"), "gfs": self._catalog("gfs")}
        features, metadata = build_raw_grid_wind_features(
            self._raw("ldaps"), self._raw("gfs"), catalog
        )
        self.assertEqual(features.shape, (3, EXPECTED_FEATURE_COUNT))
        self.assertEqual(metadata["feature_count"], EXPECTED_FEATURE_COUNT)
        self.assertEqual(
            float(features.iloc[1]["ldaps__grid_16__heightAboveGround_5_YBLWS"]),
            1167.0,
        )
        self.assertEqual(
            float(features.iloc[2]["gfs__grid_09__planetaryBoundaryLayer_0_v"]),
            2097.0,
        )
        self.assertEqual(len(set(features.columns)), EXPECTED_FEATURE_COUNT)

    def test_catalog_drift_is_rejected(self) -> None:
        catalog = {"ldaps": self._catalog("ldaps"), "gfs": self._catalog("gfs")}
        catalog["ldaps"][0]["longitude"] = 999.0
        with self.assertRaisesRegex(ValueError, "coordinate changed"):
            build_raw_grid_wind_features(self._raw("ldaps"), self._raw("gfs"), catalog)

    def test_extra_nonregistered_weather_column_is_rejected(self) -> None:
        ldaps = self._raw("ldaps")
        ldaps["surface_0_sp"] = 100000.0
        catalog = {"ldaps": self._catalog("ldaps"), "gfs": self._catalog("gfs")}
        with self.assertRaisesRegex(ValueError, "only the registered"):
            build_raw_grid_wind_features(ldaps, self._raw("gfs"), catalog)


class RawGridWindProtocolTests(unittest.TestCase):
    def test_preregister_hash_and_candidate_contract_are_exact(self) -> None:
        path = Path("configs/raw_grid_wind_lgb_preregister_v1.json")
        self.assertEqual(sha256_file(path), runner.PREREGISTER_SHA256)
        payload = runner._verify_preregister(path)
        self.assertEqual(payload["feature_contract"]["ordered_feature_count"], 212)
        self.assertEqual(
            payload["model_family"]["objectives_in_fixed_order"], ["q07", "l1"]
        )
        self.assertEqual(
            payload["candidate_family"]["fixed_blend_weights"], [0.05, 0.1, 0.2]
        )

    def test_fit_must_end_before_application(self) -> None:
        fit = pd.date_range("2022-01-01", periods=3, freq="h")
        application = pd.date_range("2022-01-01 03:00", periods=2, freq="h")
        evidence = assert_fit_before_apply(fit, application)
        self.assertTrue(evidence["fit_end_before_application_start"])
        with self.assertRaisesRegex(ValueError, "not strict-forward"):
            assert_fit_before_apply(fit, pd.date_range("2022-01-01 02:00", periods=2, freq="h"))

    def test_training_only_median_and_schema_guard(self) -> None:
        index = pd.date_range("2022-01-01", periods=180, freq="h")
        x = np.linspace(0.0, 1.0, len(index))
        features = pd.DataFrame(
            {
                "a": x,
                "b": np.sin(8.0 * x),
                "c": np.where(np.arange(len(index)) == 4, np.nan, x**2),
            },
            index=index,
        )
        target = pd.Series(0.1 + 0.8 * x, index=index)
        model = RawGridWindRegressor("l1").fit(features, target)
        prediction = model.predict(features.iloc[-8:])
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreaterEqual(float(prediction.min()), 0.0)
        with self.assertRaisesRegex(ValueError, "schema/order changed"):
            model.predict(features.iloc[-8:][["b", "a", "c"]])

    def test_weight_zero_is_value_bit_identical(self) -> None:
        index = pd.date_range("2023-01-01", periods=4, freq="h")
        baseline = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
        raw = pd.Series([0.3, 0.4, 0.5, 0.6], index=index)
        result = blend_raw_prediction(
            baseline, raw, weight=0.0, capacity_kwh=21600.0
        )
        self.assertEqual(
            np.ascontiguousarray(result.to_numpy()).tobytes(),
            np.ascontiguousarray(baseline.to_numpy()).tobytes(),
        )

    def test_selection_requires_every_slice_and_uses_registered_ties(self) -> None:
        slices = ("full", "H1")

        def comparison(delta: float) -> dict[str, object]:
            return {
                "baseline": {"score": 0.5},
                "candidate": {"score": 0.5 + delta},
                "delta": (0.5 + delta) - 0.5,
            }

        comparisons = {
            key: {name: comparison(-0.01) for name in slices} for key in CANDIDATE_KEYS
        }
        selected, audit = select_group_candidate(comparisons, slices)
        self.assertIsNone(selected)
        self.assertEqual(audit["selected"], "identity")

        for key in ("q07_w05", "l1_w05"):
            comparisons[key] = {name: comparison(0.01) for name in slices}
        selected, _ = select_group_candidate(comparisons, slices)
        self.assertEqual(selected, "q07_w05")

        comparisons["q07_w05"]["H1"] = comparison(-0.001)
        selected, _ = select_group_candidate(comparisons, slices)
        self.assertEqual(selected, "l1_w05")

    def test_json_sorted_roundtrip_uses_registered_candidate_and_slice_order(self) -> None:
        slices = ("full", "H1", "H2")

        def comparison(delta: float) -> dict[str, object]:
            return {
                "baseline": {"score": 0.5},
                "candidate": {"score": 0.5 + delta},
                "delta": (0.5 + delta) - 0.5,
            }

        comparisons = {
            key: {name: comparison(0.01 if key == "q07_w05" else -0.01) for name in slices}
            for key in CANDIDATE_KEYS
        }
        roundtripped = json.loads(json.dumps(comparisons, sort_keys=True))
        self.assertNotEqual(tuple(roundtripped), CANDIDATE_KEYS)
        selected, audit = select_group_candidate(roundtripped, slices)
        self.assertEqual(selected, "q07_w05")
        self.assertEqual(tuple(audit["candidates"]), CANDIDATE_KEYS)

    def test_stage2_identity_opens_no_2024_reader(self) -> None:
        lock = {
            "passed_groups": [],
            "locked_candidates": {group: "identity" for group in runner.TARGET_COLS},
        }
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "stage1_promotion_lock.json").write_text(
                "{}\n", encoding="utf-8"
            )
            forbidden = AssertionError("a 2024 reader was called")
            with (
                mock.patch.object(runner, "_load_stage1_lock", return_value=(lock, {})),
                mock.patch.object(runner, "_read_weather_features", side_effect=forbidden),
                mock.patch.object(runner, "_read_label_prefix", side_effect=forbidden),
                mock.patch.object(runner, "_read_full_labels", side_effect=forbidden),
                mock.patch.object(runner, "_assert_file_identity", side_effect=forbidden),
            ):
                result = runner._stage2(
                    raw_dir=Path("unused"),
                    artifact_root=Path("unused"),
                    out_dir=out_dir,
                    preregister={},
                )
            self.assertFalse(result["2024_read"])
            self.assertFalse(result["2025_read"])
            persisted = json.loads(
                (out_dir / "stage2_results.json").read_text(encoding="utf-8")
            )
            self.assertFalse(persisted["2024_weather_read"])
            self.assertFalse(persisted["2024_label_read"])

    def test_application_label_read_is_source_ordered_after_prescore_lock(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        stage1 = source[source.index("def _stage1(") : source.index("def _load_stage1_lock(")]
        self.assertLess(
            stage1.index('global_lock_path = out_dir / "stage1_global_prescore_lock.json"'),
            stage1.index("score_labels, score_label_evidence = _read_label_prefix("),
        )
        stage2 = source[source.index("def _stage2(") : source.index("def _load_stage2_lock(")]
        self.assertLess(
            stage2.index('prescore_lock_path = out_dir / "stage2_global_prescore_lock.json"'),
            stage2.index("score_labels, score_label_evidence = _read_full_labels("),
        )

    def test_final_nonpromotion_opens_no_2025_or_sample_reader(self) -> None:
        lock = {
            "all_three_groups_promoted": False,
            "stage2_passed_groups": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "stage2_promotion_lock.json").write_text(
                "{}\n", encoding="utf-8"
            )
            forbidden = AssertionError("a 2025 or sample reader was called")
            with (
                mock.patch.object(runner, "_load_stage2_lock", return_value=(lock, {})),
                mock.patch.object(runner, "_read_weather_features", side_effect=forbidden),
                mock.patch.object(runner, "_read_full_labels", side_effect=forbidden),
                mock.patch.object(runner, "_assert_file_identity", side_effect=forbidden),
                mock.patch.object(runner.pd, "read_csv", side_effect=forbidden),
            ):
                result = runner._final(
                    raw_dir=Path("unused"),
                    artifact_root=Path("unused"),
                    out_dir=out_dir,
                    preregister={},
                )
            self.assertFalse(result["2025_weather_read"])
            self.assertFalse(result["sample_submission_read"])
            self.assertFalse(result["submission_created"])


if __name__ == "__main__":
    unittest.main()

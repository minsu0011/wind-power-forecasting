from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from scripts import run_covariate_shift_feature_pruning as v1
from scripts import run_covariate_shift_feature_pruning_v2 as runner
from src.features import (
    AVAILABLE_COL,
    COORD_COLS,
    GRID_COL,
    SOURCE_VARIABLES,
    TIME_COL,
)
from src.manifest import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]


class CovariateShiftFeaturePruningV2Tests(unittest.TestCase):
    def test_v2_preregister_hash_supersession_and_incident_are_exact(self) -> None:
        path = PROJECT_DIR / "configs/covariate_shift_feature_pruning_preregister_v2.json"
        self.assertEqual(sha256_file(path), runner.PREREGISTER_SHA256)
        payload, base = runner._verify_preregister(path)
        self.assertEqual(payload["supersession"]["base_preregister_sha256"], v1.PREREGISTER_SHA256)
        self.assertEqual(base["experiment_id"], "covariate_shift_feature_pruning_strict_forward_v1")
        self.assertTrue(payload["supersession"]["v1_is_ineligible_for_selection"])
        self.assertTrue(payload["supersession"]["no_retune_or_reselection_from_v1_results"])

    def test_v2_candidate_grid_and_parameters_are_identical_to_v1(self) -> None:
        payload = json.loads(
            (PROJECT_DIR / "configs/covariate_shift_feature_pruning_preregister_v2.json").read_text(encoding="utf-8")
        )
        fixed = payload["fixed_candidate_contract"]
        self.assertEqual(tuple(fixed["keep_fractions_in_order"]), v1.KEEP_FRACTIONS)
        self.assertEqual(tuple(fixed["objectives_in_order"]), v1.OBJECTIVES)
        self.assertEqual(tuple(fixed["blend_weights_in_order"]), v1.BLEND_WEIGHTS)
        self.assertEqual(fixed["model_common_parameters"], dict(v1.MODEL_COMMON_PARAMETERS))
        self.assertEqual(fixed["candidate_count_per_group"], len(v1.CANDIDATE_KEYS))

    def test_full_weather_usecols_hashes_match_frozen_source_variables(self) -> None:
        payload = json.loads(
            (PROJECT_DIR / "configs/covariate_shift_feature_pruning_preregister_v2.json").read_text(encoding="utf-8")
        )
        for source in ("ldaps", "gfs"):
            names = runner._weather_usecols(source)
            observed = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()
            spec = payload["stage1_raw_weather_prefixes"][source]
            self.assertEqual(len(names), spec["ordered_usecols_count"])
            self.assertEqual(observed, spec["ordered_usecols_sha256"])
            self.assertEqual(tuple(names[5:]), tuple(SOURCE_VARIABLES[source]))

    def test_bounded_weather_parser_returns_zero_suffix_and_no_future_values(self) -> None:
        source = "ldaps"
        columns = runner._weather_usecols(source)
        first = ["2023-12-31 23:00:00", "2023-12-31 00:00:00", "1", "37.2", "128.9"]
        first.extend(str(position + 0.25) for position in range(len(SOURCE_VARIABLES[source])))
        future = ["2024-01-01 01:00:00", "2024-01-01 00:00:00", "1", "999999", "999999"]
        future.extend("999999" for _ in SOURCE_VARIABLES[source])
        header = ",".join(columns) + "\n"
        prefix = (header + ",".join(first) + "\n").encode("utf-8")
        whole = prefix + (",".join(future) + "\n").encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            train = raw_dir / "train"
            train.mkdir()
            path = train / "ldaps_train.csv"
            path.write_bytes(whole)
            spec = {
                "data_rows": 1,
                "bytes": len(prefix),
                "sha256": hashlib.sha256(prefix).hexdigest(),
                "end": "2023-12-31 23:00:00",
                "next_row_first_field_only": "2024-01-01 01:00:00",
            }
            frame, evidence = runner._read_physical_weather_prefix(
                raw_dir, {"stage1_raw_weather_prefixes": {source: spec}}, source
            )
        self.assertEqual(len(frame), 1)
        self.assertFalse((frame.astype(str) == "999999").any().any())
        self.assertEqual(evidence["bytes_returned_to_parser"], len(prefix))
        self.assertEqual(evidence["underlying_position_after_read"], len(prefix))
        self.assertEqual(evidence["suffix_bytes_returned_to_weather_parser"], 0)
        self.assertEqual(evidence["future_weather_value_cells_materialized"], 0)
        self.assertEqual(evidence["next_row_probe_value_bytes_read"], 0)

    def test_raw_stage1_builder_has_no_cache_open_path(self) -> None:
        source = inspect.getsource(runner._build_stage1_features)
        self.assertIn("_read_physical_weather_prefix", source)
        self.assertIn("WeatherFeatureBuilder", source)
        self.assertNotIn("read_parquet", source)
        self.assertNotIn("artifact_root", source)

    def test_postlock_cache_audit_refuses_before_touching_cache(self) -> None:
        touched = False

        def forbidden(*args: object, **kwargs: object) -> None:
            nonlocal touched
            touched = True
            raise AssertionError("cache identity touched")

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "_assert_file_identity", side_effect=forbidden
        ):
            root = Path(directory)
            with self.assertRaisesRegex(AssertionError, "identity lock does not exist"):
                runner._postlock_cache_audit(
                    artifact_root=root,
                    out_dir=root / "out",
                    preregister={},
                    raw_features={},
                    identity_lock_path=root / "missing-lock.json",
                    output_paths=[],
                )
        self.assertFalse(touched)

    def test_run_order_places_identity_lock_before_cache_audit_and_labels(self) -> None:
        source = inspect.getsource(runner._run_stage1)
        lock_position = source.index('identity_lock_path = out_dir / "stage1_candidate_identity_lock.json"')
        cache_position = source.index("cache_audit_path = _postlock_cache_audit")
        score_position = source.index("score_labels, score_label_evidence")
        self.assertLess(lock_position, cache_position)
        self.assertLess(cache_position, score_position)

    def test_frame_hash_is_bit_sensitive_and_reproduction_refuses_without_v2_metrics(self) -> None:
        index = pd.date_range("2023-01-01", periods=3, freq="h", name=TIME_COL)
        frame = pd.DataFrame({"x": np.array([1, 2, 3], dtype=np.float32)}, index=index)
        self.assertEqual(runner._frame_value_sha256(frame), runner._frame_value_sha256(frame.copy()))
        changed = frame.copy()
        changed.iloc[1, 0] = np.nextafter(np.float32(2), np.float32(3))
        self.assertNotEqual(runner._frame_value_sha256(frame), runner._frame_value_sha256(changed))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AssertionError, "v2 metrics must exist"):
                runner._reproduction_audit(Path(directory), Path(directory) / "missing.json")


if __name__ == "__main__":
    unittest.main()

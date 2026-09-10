from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts import run_shared_q07_multiseed as multiseed
from src.features import (
    AVAILABLE_COL,
    COORD_COLS,
    GRID_COL,
    SOURCE_VARIABLES,
    TIME_COL,
)
from src.manifest import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]


class SharedQ07MultiseedTests(unittest.TestCase):
    def test_preregister_is_still_content_addressed(self) -> None:
        path = PROJECT_DIR / "configs" / "shared_q07_multiseed_preregister.json"
        self.assertEqual(sha256_file(path), multiseed.PREREGISTER_SHA256)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["single_candidate"]["candidate_count"], 1)
        self.assertEqual(payload["single_candidate"]["seeds"], [42, 7, 2026])

    def test_locked_hour_intervals_partition_without_overlap(self) -> None:
        index = pd.date_range("2023-01-01 01:00", periods=8760, freq="h")
        h1 = multiseed._interval(
            index,
            pd.Timestamp("2023-01-01 01:00"),
            pd.Timestamp("2023-07-01 00:00"),
        )
        h2 = multiseed._interval(
            index,
            pd.Timestamp("2023-07-01 01:00"),
            pd.Timestamp("2024-01-01 00:00"),
        )
        self.assertFalse(np.any(h1 & h2))
        self.assertTrue(np.all(h1 | h2))
        self.assertEqual(int(h1.sum()), 4344)
        self.assertEqual(int(h2.sum()), 4416)

    def test_group_assembly_preserves_locked_affine_clip_and_bins(self) -> None:
        recipe = json.loads(
            (PROJECT_DIR / "configs" / "train_final.v3.locked.json").read_text(
                encoding="utf-8"
            )
        )
        index = pd.date_range("2024-01-01 01:00", periods=4, freq="h")
        components = {
            name: pd.Series(
                np.array([1000.0, 6000.0, 12000.0, 25000.0]), index=index
            )
            for name in multiseed.COMPONENTS
        }
        observed = multiseed._assemble_group(recipe, "kpx_group_2", components)
        raw = np.array([1000.0, 6000.0, 12000.0, 25000.0])
        prebin = np.clip(1.05 * raw - 500.0, 0.0, 21600.0 * 1.02)
        positions = np.searchsorted(
            np.array([0.25, 0.5, 0.75]), prebin / 21600.0, side="right"
        )
        expected = np.clip(
            prebin + np.array([-400.0, -1000.0, 1200.0, -200.0])[positions],
            0.0,
            21600.0 * 1.02,
        )
        np.testing.assert_allclose(
            observed.to_numpy(), expected, rtol=0.0, atol=0.0
        )

    def test_bounded_raw_reader_exposes_no_suffix_weather_bytes(self) -> None:
        source = "gfs"
        rows_per_timestamp = multiseed.RAW_ROWS_PER_TIMESTAMP[source]
        start = pd.Timestamp("2022-01-01 01:00")
        end = start + pd.Timedelta(hours=1)
        next_timestamp = end + pd.Timedelta(hours=1)
        required = [
            TIME_COL,
            AVAILABLE_COL,
            GRID_COL,
            *COORD_COLS,
            *SOURCE_VARIABLES[source],
        ]
        rows: list[dict[str, object]] = []
        for timestamp in (start, end):
            for grid_id in range(1, rows_per_timestamp + 1):
                row: dict[str, object] = {
                    TIME_COL: timestamp,
                    AVAILABLE_COL: start,
                    GRID_COL: grid_id,
                    COORD_COLS[0]: 37.0 + grid_id / 100.0,
                    COORD_COLS[1]: 128.0 + grid_id / 100.0,
                }
                row.update(
                    {
                        column: float(grid_id)
                        for column in SOURCE_VARIABLES[source]
                    }
                )
                rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bounded.csv"
            pd.DataFrame(rows, columns=required).to_csv(path, index=False)
            with path.open("a", encoding="utf-8", newline="") as stream:
                forbidden = [next_timestamp.isoformat(sep=" ")]
                forbidden.extend(["FORBIDDEN_FUTURE_VALUE"] * (len(required) - 1))
                stream.write(",".join(forbidden) + "\n")
            frame, evidence = multiseed._read_bounded_weather_csv(
                path,
                source=source,
                expected_timestamps=2,
                expected_start=start,
                expected_end=end,
                expected_next=next_timestamp,
            )
            self.assertEqual(len(frame), 2 * rows_per_timestamp)
            self.assertEqual(evidence["suffix_bytes_exposed_to_parser"], 0)
            self.assertEqual(
                evidence["underlying_file_position_after_parse"],
                evidence["physical_byte_limit"],
            )
            self.assertLess(evidence["physical_byte_limit"], path.stat().st_size)
            self.assertEqual(
                evidence["next_boundary_timestamp_only"],
                next_timestamp.isoformat(),
            )

    @staticmethod
    def _failing_comparisons() -> dict[str, object]:
        comparisons: dict[str, object] = {}
        for group, slices in multiseed.STAGE1_REQUIRED_SLICES.items():
            group_values: dict[str, object] = {}
            for slice_name in slices:
                baseline_score = 0.5
                candidate_score = 0.4
                group_values[slice_name] = {
                    "baseline": {"score": baseline_score},
                    "candidate": {"score": candidate_score},
                    "delta": candidate_score - baseline_score,
                }
            comparisons[group] = group_values
        return comparisons

    def test_tampered_locked_groups_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            source_path = out_dir / "source.py"
            input_path = out_dir / "input.bin"
            source_path.write_text("source\n", encoding="utf-8")
            input_path.write_bytes(b"input\n")
            provenance = {"runner": multiseed._snapshot_file(source_path)}
            inputs = {"bounded": multiseed._snapshot_file(input_path)}
            comparisons = self._failing_comparisons()
            result = {
                "candidate_count": 1,
                "candidate": multiseed.CANDIDATE_DESCRIPTION,
                "lock_rule": multiseed.LOCK_RULE,
                "comparisons": comparisons,
                "comparisons_sha256": multiseed._canonical_sha256(comparisons),
                "locked_groups": [],
                "provenance_before": provenance,
                "provenance_after": provenance,
                "provenance_snapshot_sha256": multiseed._canonical_sha256(
                    provenance
                ),
                "provenance_sha256": multiseed._provenance_sha_map(provenance),
                "stage1_input_snapshot_before": inputs,
                "stage1_input_snapshot_after": inputs,
                "stage1_input_snapshot_sha256": multiseed._canonical_sha256(inputs),
            }
            result_path = out_dir / "stage1_results.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            lock = {
                "preregister_sha256": multiseed.PREREGISTER_SHA256,
                "stage1_results_sha256": sha256_file(result_path),
                "locked_groups": ["kpx_group_1"],
                "comparisons_sha256": result["comparisons_sha256"],
                "selection_rule": multiseed.LOCK_RULE,
                "provenance_snapshot_sha256": result[
                    "provenance_snapshot_sha256"
                ],
                "provenance_sha256": result["provenance_sha256"],
                "stage1_input_snapshot_sha256": result[
                    "stage1_input_snapshot_sha256"
                ],
                "stage2_candidate_frozen": True,
                "no_2024_reselection": True,
            }
            (out_dir / "stage1_lock.json").write_text(
                json.dumps(lock), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                AssertionError, "locked_groups differs from result/rule"
            ):
                multiseed._load_stage1_lock(out_dir)

    def test_no_locked_groups_skip_stage2_before_2024_reads(self) -> None:
        lock = {
            "locked_groups": [],
            "provenance_snapshot_sha256": "provenance",
            "provenance_sha256": {"runner": "sha"},
            "stage1_input_snapshot_sha256": "inputs",
        }
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory) / "out"
            out_dir.mkdir()
            (out_dir / "stage1_lock.json").write_text("{}", encoding="utf-8")
            forbidden = AssertionError("2024 read attempted")
            with (
                mock.patch.object(multiseed, "_load_stage1_lock", return_value=lock),
                mock.patch.object(multiseed, "_read_labels", side_effect=forbidden) as labels,
                mock.patch.object(multiseed, "_read_features", side_effect=forbidden) as features,
                mock.patch.object(
                    multiseed, "_read_prediction", side_effect=forbidden
                ) as predictions,
            ):
                promotion = multiseed._stage2(
                    raw_dir=Path(directory) / "raw",
                    cache_dir=Path(directory) / "cache",
                    artifact_root=Path(directory) / "artifacts",
                    out_dir=out_dir,
                    recipe={},
                )
            self.assertEqual(promotion["promoted_groups"], [])
            labels.assert_not_called()
            features.assert_not_called()
            predictions.assert_not_called()
            result = json.loads(
                (out_dir / "stage2_results.json").read_text(encoding="utf-8")
            )
            self.assertFalse(result["2024_read"])

    def test_no_promotion_manifest_binds_all_required_source_hashes(self) -> None:
        required = {
            "runner",
            "features",
            "metric",
            "manifest",
            "test",
            "preregister",
            "recipe",
        }
        recipe_path = PROJECT_DIR / "configs" / "train_final.v3.locked.json"
        preregister_path = (
            PROJECT_DIR / "configs" / "shared_q07_multiseed_preregister.json"
        )
        provenance = multiseed._snapshot_named_files(
            multiseed._provenance_paths(
                recipe_path=recipe_path, preregister_path=preregister_path
            )
        )
        self.assertEqual(set(provenance), required)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out_dir = root / "out"
            out_dir.mkdir()
            input_path = root / "bounded-input"
            input_path.write_bytes(b"bounded")
            input_snapshot = {"bounded": multiseed._snapshot_file(input_path)}
            stage1_result = {
                "provenance_before": provenance,
                "provenance_snapshot_sha256": multiseed._canonical_sha256(
                    provenance
                ),
                "provenance_sha256": multiseed._provenance_sha_map(provenance),
                "stage1_input_snapshot_before": input_snapshot,
                "stage1_input_snapshot_sha256": multiseed._canonical_sha256(
                    input_snapshot
                ),
                "snapshots_unchanged_during_stage1": True,
            }
            (out_dir / "stage1_results.json").write_text(
                json.dumps(stage1_result), encoding="utf-8"
            )
            (out_dir / "stage1_lock.json").write_text("{}", encoding="utf-8")
            (out_dir / "postlock_bit_exact_audit.json").write_text(
                "{}", encoding="utf-8"
            )
            stage2_result = {
                "promoted_groups": [],
                "provenance_snapshot_sha256": stage1_result[
                    "provenance_snapshot_sha256"
                ],
                "provenance_sha256": stage1_result["provenance_sha256"],
            }
            stage2_path = out_dir / "stage2_results.json"
            stage2_path.write_text(json.dumps(stage2_result), encoding="utf-8")
            promotion = {
                "preregister_sha256": multiseed.PREREGISTER_SHA256,
                "stage2_results_sha256": sha256_file(stage2_path),
                "promoted_groups": [],
                "provenance_snapshot_sha256": stage1_result[
                    "provenance_snapshot_sha256"
                ],
                "provenance_sha256": stage1_result["provenance_sha256"],
            }
            (out_dir / "stage2_promotion_lock.json").write_text(
                json.dumps(promotion), encoding="utf-8"
            )
            manifest = multiseed._finalize(
                raw_dir=root / "raw",
                cache_dir=root / "cache",
                artifact_root=root / "artifacts",
                out_dir=out_dir,
                recipe={},
                recipe_path=recipe_path,
                preregister_path=preregister_path,
            )
            self.assertEqual(set(manifest["provenance_sha256"]), required)
            manifest_input_hashes = {
                Path(record["path"]).resolve(): record.get("sha256")
                for record in manifest["inputs"]
                if record.get("sha256") is not None
            }
            for record in provenance.values():
                self.assertEqual(
                    manifest_input_hashes[Path(record["path"]).resolve()],
                    record["sha256"],
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts import run_ficr_bayes_decision_strict as bayes
from src.manifest import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]


class StrictFICRBayesDecisionTests(unittest.TestCase):
    def test_preregister_hash_candidate_family_and_config_are_exact(self) -> None:
        path = PROJECT_DIR / "configs" / "ficr_bayes_decision_preregister.json"
        self.assertEqual(sha256_file(path), bayes.PREREGISTER_SHA256)
        payload, config = bayes._verify_preregister(path)
        self.assertEqual(payload["candidate_family"]["candidate_count"], 4)
        self.assertEqual(
            tuple(payload["candidate_family"]["methods_in_complexity_order"]),
            bayes.METHODS,
        )
        self.assertEqual(config.random_state, 42)
        self.assertEqual(config.decision_shrinkage, 0.65)

    def test_every_stage1_application_is_strictly_after_fit(self) -> None:
        contract = bayes._stage1_rolling_contract()
        for _, fit_index, application_index in (
            *contract["g12_directions"],
            *contract["g3_directions"],
        ):
            self.assertLess(fit_index.max(), application_index.min())
            self.assertEqual(len(fit_index.intersection(application_index)), 0)
        g12_application = contract["g12_application"]
        g12_union = contract["g12_segments"]["Q2"].append(
            [
                contract["g12_segments"]["Q3"],
                contract["g12_segments"]["Q4"],
            ]
        )
        self.assertTrue(g12_union.equals(g12_application))
        g3_union = contract["g3_directions"][0][2]
        for _, _, application in contract["g3_directions"][1:]:
            g3_union = g3_union.append(application)
        self.assertTrue(g3_union.equals(contract["g3_application"]))

    def test_bounded_label_parser_receives_no_suffix_bytes(self) -> None:
        index = pd.date_range("2022-01-01 01:00", periods=3, freq="h")
        frame = pd.DataFrame(
            {
                "kst_dtm": index,
                "kpx_group_1": [1.0, 2.0, 999999.0],
                "kpx_group_2": [3.0, 4.0, 999999.0],
                "kpx_group_3": [5.0, 6.0, 999999.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
            raw = path.read_bytes()
            newline_positions = [i for i, value in enumerate(raw) if value == 10]
            prefix_bytes = newline_positions[2] + 1
            prefix_sha = hashlib.sha256(raw[:prefix_bytes]).hexdigest()
            expected_index = pd.DatetimeIndex(index[:2], name="forecast_kst_dtm")
            labels, evidence = bayes._read_bounded_labels(
                path,
                data_rows=2,
                byte_limit=prefix_bytes,
                expected_prefix_sha256=prefix_sha,
                expected_index=expected_index,
            )
            self.assertEqual(len(labels), 2)
            self.assertNotIn(999999.0, labels.to_numpy())
            self.assertEqual(evidence["suffix_bytes_exposed_to_parser"], 0)
            self.assertEqual(
                evidence["underlying_file_position_after_read"], prefix_bytes
            )
            self.assertLess(prefix_bytes, path.stat().st_size)

    def test_prediction_reader_rejects_persisted_index_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.parquet"
            frame = pd.DataFrame(
                {group: [1.0, 2.0] for group in bayes.TARGET_COLS},
                index=pd.date_range("2023-01-02", periods=2, freq="h"),
            )
            frame.to_parquet(path)
            expected = pd.date_range(
                "2023-01-01", periods=2, freq="h", name="forecast_kst_dtm"
            )
            with self.assertRaisesRegex(AssertionError, "timestamp index"):
                bayes._read_prediction(
                    path, expected, required_columns=bayes.TARGET_COLS
                )

    @staticmethod
    def _comparisons() -> dict[str, object]:
        comparisons: dict[str, object] = {}
        for group, segments in bayes.STAGE1_REQUIRED_SEGMENTS.items():
            group_methods: dict[str, object] = {}
            for method in bayes.METHODS:
                group_methods[method] = {}
                for segment in segments:
                    baseline = 0.50
                    candidate = 0.49
                    group_methods[method][segment] = {
                        "baseline": {"score": baseline},
                        "candidate": {"score": candidate},
                        "delta": candidate - baseline,
                    }
            comparisons[group] = group_methods
        return comparisons

    @staticmethod
    def _set_method_delta(
        comparisons: dict[str, object], group: str, method: str, delta: float
    ) -> None:
        for segment in bayes.STAGE1_REQUIRED_SEGMENTS[group]:
            record = comparisons[group][method][segment]
            record["candidate"]["score"] = 0.50 + delta
            record["delta"] = record["candidate"]["score"] - 0.50

    def test_selection_requires_every_segment_and_maximises_minimum(self) -> None:
        comparisons = self._comparisons()
        self._set_method_delta(comparisons, "kpx_group_1", "global_conformal", 0.002)
        self._set_method_delta(comparisons, "kpx_group_1", "binned_kde", 0.003)
        self._set_method_delta(comparisons, "kpx_group_2", "quantile_stack", 0.004)
        comparisons["kpx_group_2"]["quantile_stack"]["Q4"]["candidate"][
            "score"
        ] = 0.50
        comparisons["kpx_group_2"]["quantile_stack"]["Q4"]["delta"] = 0.0
        recipe, diagnostics = bayes._select_recipe(comparisons)
        self.assertEqual(recipe["kpx_group_1"], "binned_kde")
        self.assertEqual(recipe["kpx_group_2"], "identity")
        self.assertEqual(recipe["kpx_group_3"], "identity")
        self.assertFalse(diagnostics["kpx_group_2"]["selection_uses_2024"])

    def test_global_stage2_promotion_requires_every_selected_segment(self) -> None:
        recipe = {
            "kpx_group_1": "binned_kde",
            "kpx_group_2": "identity",
            "kpx_group_3": "identity",
        }
        comparisons = {
            "kpx_group_1": {
                segment: {
                    "baseline": {"score": 0.5},
                    "candidate": {"score": 0.501},
                    "delta": 0.501 - 0.5,
                }
                for segment in bayes.STAGE2_REQUIRED_SEGMENTS
            }
        }
        promoted, _ = bayes._stage2_promoted(comparisons, recipe)
        self.assertTrue(promoted)
        comparisons["kpx_group_1"]["H2"]["candidate"]["score"] = 0.5
        comparisons["kpx_group_1"]["H2"]["delta"] = 0.0
        promoted, _ = bayes._stage2_promoted(comparisons, recipe)
        self.assertFalse(promoted)

    def test_identity_recipe_skips_every_2024_reader(self) -> None:
        recipe = {group: "identity" for group in bayes.TARGET_COLS}
        lock = {
            "locked_recipe": recipe,
            "stage1_results_sha256": "stage1-result",
        }
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "stage1_recipe_lock.json").write_text(
                "{}", encoding="utf-8"
            )
            forbidden = AssertionError("2024 reader called")
            with (
                mock.patch.object(
                    bayes, "_load_stage1_lock", return_value=(lock, {})
                ),
                mock.patch.object(
                    bayes, "_stage2_input_snapshot", side_effect=forbidden
                ) as snapshot,
                mock.patch.object(
                    bayes, "_read_full_labels", side_effect=forbidden
                ) as labels,
                mock.patch.object(
                    bayes, "_load_period_sources", side_effect=forbidden
                ) as predictions,
            ):
                observed = bayes._stage2(
                    raw_dir=out_dir / "raw",
                    artifact_root=out_dir / "artifacts",
                    out_dir=out_dir,
                    config=bayes.ProbabilisticDecisionConfig(),
                )
            self.assertFalse(observed["recipe_promoted"])
            snapshot.assert_not_called()
            labels.assert_not_called()
            predictions.assert_not_called()
            result = json.loads(
                (out_dir / "stage2_results.json").read_text(encoding="utf-8")
            )
            self.assertFalse(result["2024_read"])

    def test_stage1_lock_recomputes_recipe_and_rejects_tampering(self) -> None:
        comparisons = self._comparisons()
        recipe, selection = bayes._select_recipe(comparisons)
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            source = out_dir / "source.txt"
            bounded = out_dir / "input.txt"
            source.write_text("source", encoding="utf-8")
            bounded.write_text("input", encoding="utf-8")
            provenance = {"source": bayes._snapshot_file(source)}
            inputs = {"input": bayes._snapshot_file(bounded)}
            result = {
                "candidate_methods": list(bayes.METHODS),
                "candidate_count": len(bayes.METHODS),
                "comparisons": comparisons,
                "comparisons_sha256": bayes._canonical_sha256(comparisons),
                "selection": selection,
                "locked_recipe": recipe,
                "locked_recipe_sha256": bayes._canonical_sha256(recipe),
                "provenance_before": provenance,
                "provenance_snapshot_sha256": bayes._canonical_sha256(provenance),
                "stage1_input_snapshot_before": inputs,
                "stage1_input_snapshot_sha256": bayes._canonical_sha256(inputs),
            }
            bayes._write_json(out_dir / "stage1_results.json", result)
            lock = {
                "preregister_sha256": bayes.PREREGISTER_SHA256,
                "stage1_results_sha256": sha256_file(
                    out_dir / "stage1_results.json"
                ),
                "comparisons_sha256": result["comparisons_sha256"],
                "locked_recipe": recipe,
                "locked_recipe_sha256": result["locked_recipe_sha256"],
                "provenance_snapshot_sha256": result[
                    "provenance_snapshot_sha256"
                ],
                "stage1_input_snapshot_sha256": result[
                    "stage1_input_snapshot_sha256"
                ],
                "candidate_and_hyperparameters_frozen": True,
                "no_2024_reselection_or_retuning": True,
            }
            bayes._write_json(out_dir / "stage1_recipe_lock.json", lock)
            bayes._load_stage1_lock(out_dir)
            tampered = copy.deepcopy(lock)
            tampered["locked_recipe"]["kpx_group_1"] = "binned_kde"
            (out_dir / "stage1_recipe_lock.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            with self.assertRaisesRegex(AssertionError, "recomputed rule"):
                bayes._load_stage1_lock(out_dir)


if __name__ == "__main__":
    unittest.main()

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.metric import CAPACITY_KWH
from src.pseudo_transfer import (
    ActualAccessLedger,
    GROUP1,
    GROUP2,
    GROUP3,
    MapperConfig,
    build_weather_training,
    build_weather_training_strict,
    make_pseudo_labels,
    make_pseudo_labels_strict,
    read_bounded_actuals,
)


class PseudoTransferLeakageTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2022-01-01 01:00", periods=480, freq="h")
        angle = np.linspace(0.0, 12.0, len(self.index))
        cf1 = np.clip(0.45 + 0.25 * np.sin(angle), 0.0, 1.0)
        cf2 = np.clip(0.40 + 0.20 * np.cos(angle), 0.0, 1.0)
        cf3 = np.clip(0.10 + 0.55 * cf1 + 0.25 * cf2, 0.0, 1.0)
        self.labels = pd.DataFrame(
            {
                GROUP1: cf1 * CAPACITY_KWH[GROUP1],
                GROUP2: cf2 * CAPACITY_KWH[GROUP2],
                GROUP3: cf3 * CAPACITY_KWH[GROUP3],
            },
            index=self.index,
        )

    def test_mapper_never_accepts_validation_overlap(self):
        with self.assertRaisesRegex(ValueError, "validation/test actual"):
            make_pseudo_labels(
                self.labels,
                mapper_train_index=self.index[:240],
                pseudo_index=self.index[240:360],
                forbidden_validation_index=self.index[200:280],
                config=MapperConfig(min_samples_leaf=20),
            )

    def test_pseudo_audit_records_only_allowed_actual_uses(self):
        result = make_pseudo_labels(
            self.labels,
            mapper_train_index=self.index[240:400],
            pseudo_index=self.index[:200],
            forbidden_validation_index=self.index[400:],
            config=MapperConfig(min_samples_leaf=20),
        )
        self.assertEqual(result.pseudo_cf.index.tolist(), self.index[:200].tolist())
        self.assertTrue(result.pseudo_cf.notna().all())
        self.assertGreaterEqual(float(result.pseudo_cf.min()), 0.0)
        self.assertLessEqual(float(result.pseudo_cf.max()), 1.02)
        self.assertFalse(
            result.audit["validation_or_test_g1g2_values_accessed"]
        )
        self.assertEqual(result.audit["forbidden_validation_rows"], 80)

    def test_weather_rows_remain_actual_then_pseudo_without_sorting(self):
        weather = pd.DataFrame(
            {"g3_weather": np.linspace(0.0, 1.0, len(self.index))},
            index=self.index,
        )
        actual_index = self.index[240:400]
        pseudo_index = self.index[:200]
        pseudo = pd.Series(0.4, index=pseudo_index)
        bundle = build_weather_training(
            weather,
            self.labels[GROUP3],
            pseudo,
            actual_index=actual_index,
            pseudo_index=pseudo_index,
        )
        self.assertTrue(bundle.ordered_index[: len(bundle.actual_index)].equals(bundle.actual_index))
        self.assertTrue(bundle.ordered_index[len(bundle.actual_index) :].equals(bundle.pseudo_index))
        self.assertGreater(bundle.ordered_index[0], bundle.ordered_index[-1])
        self.assertTrue(bundle.features.index.equals(bundle.ordered_index))
        np.testing.assert_allclose(bundle.sample_weight, 1.0)
        np.testing.assert_allclose(
            bundle.target_cf[-len(bundle.pseudo_index) :], 0.4
        )

    def test_target_like_weather_column_is_rejected(self):
        weather = pd.DataFrame(
            {GROUP1: np.ones(len(self.index))}, index=self.index
        )
        with self.assertRaisesRegex(ValueError, "target/SCADA-like"):
            build_weather_training(
                weather,
                self.labels[GROUP3],
                pd.Series(0.4, index=self.index[:200]),
                actual_index=self.index[240:400],
                pseudo_index=self.index[:200],
            )

    def test_strict_bounded_reader_records_zero_forbidden_g12_rows(self):
        source = self.labels.copy()
        source.insert(0, "kst_dtm", source.index)
        ledger = ActualAccessLedger()
        forbidden = self.index[400:]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            source.to_csv(path, index=False)
            bounded = read_bounded_actuals(
                path,
                columns=(GROUP1, GROUP2),
                expected_index=self.index[:400],
                ledger=ledger,
                purpose="test:bounded_g12",
                forbidden_g12_index=forbidden,
            )
        self.assertTrue(bounded.index.equals(self.index[:400].rename("forecast_kst_dtm")))
        self.assertEqual(tuple(bounded.columns), (GROUP1, GROUP2))
        self.assertEqual(ledger.as_dict()["forbidden_g1g2_rows_materialized"], 0)
        self.assertEqual(ledger.events[0]["last_timestamp"], self.index[399].isoformat())
        self.assertEqual(
            ledger.events[0]["csv_reader_contract"],
            {
                "usecols": ["kst_dtm", GROUP1, GROUP2],
                "nrows": 400,
                "skip_data_rows": 0,
            },
        )

    def test_strict_reader_rejects_forbidden_request_before_csv_access(self):
        ledger = ActualAccessLedger()
        with patch("src.pseudo_transfer.pd.read_csv") as reader:
            with self.assertRaisesRegex(ValueError, "overlaps forbidden"):
                read_bounded_actuals(
                    "must_not_be_opened.csv",
                    columns=(GROUP1, GROUP2),
                    expected_index=self.index,
                    ledger=ledger,
                    purpose="test:preflight",
                    forbidden_g12_index=self.index[400:],
                )
        reader.assert_not_called()
        self.assertEqual(ledger.events, [])

    def test_strict_mapper_access_is_instrumented_and_bit_exact(self):
        mapper_index = self.index[240:400]
        pseudo_index = self.index[:200]
        forbidden = self.index[400:]
        ledger = ActualAccessLedger()
        strict = make_pseudo_labels_strict(
            self.labels.loc[mapper_index, [GROUP1, GROUP2, GROUP3]],
            self.labels.loc[pseudo_index, [GROUP1, GROUP2]],
            forbidden_validation_index=forbidden,
            config=MapperConfig(min_samples_leaf=20),
            ledger=ledger,
            audit_prefix="test",
        )
        legacy = make_pseudo_labels(
            self.labels,
            mapper_train_index=mapper_index,
            pseudo_index=pseudo_index,
            forbidden_validation_index=forbidden,
            config=MapperConfig(min_samples_leaf=20),
        )
        np.testing.assert_array_equal(
            strict.pseudo_cf.to_numpy().view(np.uint64),
            legacy.pseudo_cf.to_numpy().view(np.uint64),
        )
        self.assertEqual(ledger.as_dict()["forbidden_g1g2_rows_materialized"], 0)
        self.assertEqual(
            [event["purpose"] for event in ledger.events],
            ["test:mapper_fit", "test:pseudo_application"],
        )
        self.assertFalse(strict.audit["validation_or_test_g1g2_values_accessed"])

    def test_strict_weather_rejects_extra_validation_actuals(self):
        weather = pd.DataFrame(
            {"g3_weather": np.linspace(0.0, 1.0, len(self.index))},
            index=self.index,
        )
        actual_index = self.index[240:400]
        pseudo_index = self.index[:200]
        with self.assertRaisesRegex(ValueError, "exactly training timestamps"):
            build_weather_training_strict(
                weather,
                self.labels.loc[self.index[240:], GROUP3],
                pd.Series(0.4, index=pseudo_index),
                actual_index=actual_index,
                pseudo_index=pseudo_index,
            )


if __name__ == "__main__":
    unittest.main()

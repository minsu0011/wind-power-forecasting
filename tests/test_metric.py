import unittest

import numpy as np

from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, metric, score_details


class GroupMetricsTests(unittest.TestCase):
    def test_actual_at_ten_percent_is_included_and_forecast_is_not_filter(self):
        result = group_metrics(
            actual=np.array([9.999, 10.0]),
            forecast=np.array([1_000_000.0, 18.0]),
            capacity_kwh=100.0,
        )

        self.assertEqual(result.n_evaluated, 1)
        self.assertAlmostEqual(result.nmae, 0.08)
        self.assertAlmostEqual(result.ficr, 0.75)
        self.assertEqual(result.between_6_and_8_count, 1)

    def test_ficr_boundary_bins_are_4_3_0(self):
        result = group_metrics(
            actual=np.array([50.0, 50.0, 50.0]),
            forecast=np.array([56.0, 58.0, 58.000001]),
            capacity_kwh=100.0,
        )

        self.assertEqual(result.within_6_count, 1)
        self.assertEqual(result.between_6_and_8_count, 1)
        self.assertEqual(result.over_8_count, 1)
        self.assertAlmostEqual(result.ficr, 7.0 / 12.0)

    def test_ficr_is_weighted_by_actual_energy_not_row_count(self):
        result = group_metrics(
            actual=np.array([10.0, 90.0]),
            forecast=np.array([10.0, 99.0]),
            capacity_kwh=100.0,
        )

        self.assertAlmostEqual(result.nmae, 0.045)
        self.assertAlmostEqual(result.ficr, 0.10)
        self.assertAlmostEqual(result.within_6_energy_share, 0.10)
        self.assertAlmostEqual(result.over_8_energy_share, 0.90)

    def test_missing_actual_is_ignored_but_missing_evaluated_forecast_fails(self):
        result = group_metrics(
            actual=np.array([np.nan, 50.0]),
            forecast=np.array([np.nan, 50.0]),
            capacity_kwh=100.0,
        )
        self.assertAlmostEqual(result.ficr, 1.0)

        with self.assertRaisesRegex(ValueError, "non-finite forecasts"):
            group_metrics(
                actual=np.array([50.0]),
                forecast=np.array([np.nan]),
                capacity_kwh=100.0,
            )

    def test_invalid_shapes_and_empty_evaluation_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            group_metrics([10.0], [10.0, 20.0], 100.0)
        with self.assertRaisesRegex(ValueError, "no actual values"):
            group_metrics([0.0, 9.9], [0.0, 0.0], 100.0)


class CompetitionMetricsTests(unittest.TestCase):
    def test_groups_are_equally_weighted_despite_different_valid_counts(self):
        answer = {
            TARGET_COLS[0]: np.array([0.5 * CAPACITY_KWH[TARGET_COLS[0]], 0.0]),
            TARGET_COLS[1]: np.array([0.5 * CAPACITY_KWH[TARGET_COLS[1]]] * 2),
            TARGET_COLS[2]: np.array([0.5 * CAPACITY_KWH[TARGET_COLS[2]]] * 2),
        }
        prediction = {
            TARGET_COLS[0]: answer[TARGET_COLS[0]].copy(),
            TARGET_COLS[1]: answer[TARGET_COLS[1]]
            + 0.07 * CAPACITY_KWH[TARGET_COLS[1]],
            TARGET_COLS[2]: answer[TARGET_COLS[2]]
            + 0.09 * CAPACITY_KWH[TARGET_COLS[2]],
        }

        result = score_details(answer, prediction)

        self.assertAlmostEqual(result.one_minus_nmae, 1.0 - (0.0 + 0.07 + 0.09) / 3)
        self.assertAlmostEqual(result.ficr, (1.0 + 0.75 + 0.0) / 3)
        self.assertAlmostEqual(result.total_score, 0.765)
        self.assertEqual(
            [result.by_group[col].n_evaluated for col in TARGET_COLS],
            [1, 2, 2],
        )

        official_tuple = metric(answer, prediction)
        self.assertEqual(len(official_tuple), 3)
        np.testing.assert_allclose(
            official_tuple,
            (result.total_score, result.one_minus_nmae, result.ficr),
        )

    def test_missing_target_column_fails_loudly(self):
        answer = {column: np.array([CAPACITY_KWH[column]]) for column in TARGET_COLS}
        prediction = answer.copy()
        del prediction[TARGET_COLS[-1]]

        with self.assertRaisesRegex(ValueError, "missing target column"):
            score_details(answer, prediction)


if __name__ == "__main__":
    unittest.main()

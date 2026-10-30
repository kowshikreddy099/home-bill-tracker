"""Regression tests for conservative next-month estimates."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict_bills as forecast  # noqa: E402


def monthly(values):
    return pd.Series(values, index=pd.period_range("2024-01", periods=len(values), freq="M"))


class ForecastGuardrailTests(unittest.TestCase):
    def setUp(self):
        # Keep tests fast while exercising the two models most likely to disagree
        # after a spike. Production still evaluates every configured model.
        self.models = forecast.MODELS
        forecast.MODELS = {
            "Naive (last month)": None,
            "Ridge regression": self.models["Ridge regression"],
        }

    def tearDown(self):
        forecast.MODELS = self.models

    def test_one_off_spike_does_not_inflate_following_estimate(self):
        history = monthly([900] * 15 + [12000, 900, 900])
        prediction, _, _ = forecast.predict_next(history)
        self.assertLessEqual(prediction, forecast.forecast_upper_bound(history))
        self.assertLess(prediction, 1400)

    def test_steady_bill_stays_steady(self):
        history = monthly([899] * 18)
        prediction, _, _ = forecast.predict_next(history)
        self.assertEqual(prediction, 899)

    def test_short_history_uses_the_latest_bill(self):
        history = monthly([500, 550, 600])
        prediction, model, scores = forecast.predict_next(history)
        self.assertEqual(prediction, 600)
        self.assertEqual(model, "Naive (last month)")
        self.assertEqual(scores[model], 0)

    def test_gradual_increase_is_not_flattened(self):
        history = monthly([800 + 20 * month for month in range(18)])
        prediction, _, _ = forecast.predict_next(history)
        self.assertGreater(prediction, history.iloc[-1])
        self.assertLessEqual(prediction, forecast.forecast_upper_bound(history))

    def test_seasonal_bill_remains_within_the_expected_range(self):
        # Two yearly cycles with a summer peak and a low winter bill.
        history = monthly([600, 650, 800, 1100, 1500, 1400, 1000, 750, 650, 600, 580, 620] * 2)
        prediction, _, _ = forecast.predict_next(history)
        self.assertGreater(prediction, 500)
        self.assertLess(prediction, 850)

    def test_zero_history_is_safe(self):
        history = monthly([0] * 18)
        prediction, _, _ = forecast.predict_next(history)
        self.assertEqual(prediction, 0)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

from external_data import _fetch_usd_inr_history, indian_macro_features


class IndianMacroDataTests(unittest.TestCase):
    def test_usd_inr_rates_are_lagged_before_alignment(self):
        _fetch_usd_inr_history.cache_clear()
        rows = [
            {"date": f"2024-01-{day:02d}", "base": "USD", "quote": "INR", "rate": 82.0 + day / 10}
            for day in range(1, 11)
        ]
        response = Mock()
        response.json.return_value = rows
        response.raise_for_status.return_value = None
        index = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        with patch("external_data.requests.get", return_value=response) as request:
            features, status = indian_macro_features(index)

        self.assertIsNotNone(features)
        self.assertTrue(pd.isna(features.iloc[0]["usd_inr_rate"]))
        self.assertAlmostEqual(features.iloc[1]["usd_inr_rate"], 82.1)
        self.assertEqual(status["reference_date"], "2024-01-09")
        self.assertEqual(status["available_from"], "2024-01-10")
        self.assertEqual(status["lag_days"], 1)
        request.assert_called_once()

    def test_unavailable_external_source_returns_price_only_fallback(self):
        _fetch_usd_inr_history.cache_clear()
        index = pd.date_range("2024-02-01", periods=3, freq="D", tz="UTC")
        with patch("external_data.requests.get", side_effect=requests.Timeout("offline")):
            features, status = indian_macro_features(index)
        self.assertIsNone(features)
        self.assertFalse(status["available"])


if __name__ == "__main__":
    unittest.main()

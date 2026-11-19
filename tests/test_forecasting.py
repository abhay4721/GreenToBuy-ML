import unittest

import numpy as np
import pandas as pd
from fastapi import HTTPException

from forecasting import build_features, forecast
from market_data import market_profile, normalize_symbol, validate_market_data, yahoo_ticker
from server import get_stock_analysis, health


def sample_data(rows=340):
    index = pd.date_range("2020-01-01", periods=rows, freq="B", tz="UTC")
    rng = np.random.default_rng(23)
    close = 100 + np.cumsum(rng.normal(0.08, 1.1, rows))
    return pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.012, "low": close * 0.988,
         "close": close, "volume": rng.integers(100_000, 900_000, rows)},
        index=index,
    )


class DataPipelineTests(unittest.TestCase):
    def test_api_health_and_invalid_request_parameters(self):
        self.assertEqual(health(), {"status": "ok"})
        with self.assertRaises(HTTPException) as invalid_symbol:
            get_stock_analysis("not a ticker")
        self.assertEqual(invalid_symbol.exception.status_code, 422)
        with self.assertRaises(HTTPException) as invalid_interval:
            get_stock_analysis("AAPL", interval="weekly")
        self.assertEqual(invalid_interval.exception.status_code, 422)
        with self.assertRaises(HTTPException) as invalid_horizon:
            get_stock_analysis("AAPL", horizon=3)
        self.assertEqual(invalid_horizon.exception.status_code, 422)

    def test_symbol_normalization_and_rejection(self):
        self.assertEqual(normalize_symbol(" aapl "), "AAPL")
        self.assertEqual(yahoo_ticker("XAUUSD"), "GC=F")
        self.assertEqual(yahoo_ticker("RELIANCE.NS"), "RELIANCE.NS")
        self.assertEqual(market_profile("RELIANCE.NS"), "india")
        self.assertEqual(market_profile("EURUSD"), "forex")
        self.assertEqual(market_profile("MSFT"), "global_equity")
        with self.assertRaises(ValueError):
            normalize_symbol("bad ticker!")

    def test_data_validation_deduplicates_and_drops_invalid_prices(self):
        data = sample_data(4)
        data.iloc[1, data.columns.get_loc("close")] = np.nan
        data = pd.concat([data, data.iloc[[0]]])
        clean = validate_market_data(data)
        self.assertEqual(len(clean), 3)
        self.assertTrue(clean.index.is_monotonic_increasing)

    def test_features_use_only_current_and_prior_bars(self):
        data = sample_data(260)
        baseline = build_features(data)
        changed = data.copy()
        changed.iloc[-1, changed.columns.get_loc("close")] *= 1.5
        recalculated = build_features(changed)
        pd.testing.assert_series_equal(baseline.iloc[-2], recalculated.iloc[-2])
        self.assertIn("adx", baseline.columns)
        self.assertIn("obv", baseline.columns)

    def test_forecast_metrics_and_horizons(self):
        data = sample_data(650)
        for horizon in (1, 5, 10, 30):
            with self.subTest(horizon=horizon):
                result = forecast(data, horizon=horizon, model_choice="ridge")
                self.assertGreater(result.forecast, 0)
                self.assertEqual(result.horizon, horizon)
                self.assertGreater(result.validation_points, 0)
                self.assertIn("mae", result.metrics)
                self.assertIn("directional_accuracy", result.naive_metrics)
                self.assertLessEqual(result.uncertainty["lower"], result.uncertainty["upper"])
                self.assertEqual(
                    set(result.model_comparison),
                    {"Ridge", "Random Forest", "Extra Trees", "Hist Gradient Boosting", "SVR", "Weighted Ensemble", "Naïve: unchanged close"},
                )
                self.assertEqual(result.validation_points, 40)

    def test_weighted_ensemble_uses_validation_weights(self):
        result = forecast(sample_data(650), horizon=5, model_choice="auto", market_profile="forex")
        self.assertIn(result.model, ("Weighted Ensemble", "Naïve: unchanged close"))
        self.assertEqual(result.market_profile, "forex")
        self.assertAlmostEqual(sum(result.ensemble_weights.values()), 1.0)
        self.assertEqual(len(result.ensemble_weights), 5)

    def test_indian_market_profile_includes_indian_trend_tuning(self):
        result = forecast(sample_data(650), horizon=5, market_profile="india")
        self.assertEqual(result.market_profile, "india")
        self.assertIn("Hist Gradient Boosting", result.model_comparison)
        self.assertGreater(result.validation_points, 0)

    def test_empty_or_short_history_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "250"):
            forecast(sample_data(120))
        with self.assertRaises(ValueError):
            forecast(sample_data(), horizon=3)


if __name__ == "__main__":
    unittest.main()

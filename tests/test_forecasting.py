import unittest
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException

from forecasting import _stack_weights, build_features, forecast
from market_data import market_profile, normalize_symbol, validate_market_data, yahoo_ticker
from local_analysis import analyze_symbol
from quant_analysis import analyze_market
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
            get_stock_analysis("AAPL", horizon=2)
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

    def test_quant_diagnostics_are_finite_or_explicitly_missing(self):
        data = sample_data(260)
        diagnostics, risk = analyze_market(data, 5)
        self.assertIn("rule-based", diagnostics["regime_method"])
        self.assertIn("trailing_returns_pct", diagnostics)
        self.assertIn("price_vs_moving_average_pct", diagnostics)
        self.assertIn("historical_var_95_pct_per_bar", risk)
        self.assertIn("downside_deviation_pct_per_bar", risk)
        self.assertGreaterEqual(risk["ewma_volatility_pct_per_bar"], 0)
        json.dumps({"diagnostics": diagnostics, "risk": risk}, allow_nan=False)

    def test_stacker_weights_are_nonnegative_and_sum_to_one(self):
        rng = np.random.default_rng(8)
        actual = rng.normal(size=120)
        predictions = np.column_stack((actual + rng.normal(0, 0.1, 120), rng.normal(size=120), -actual))
        weights = _stack_weights(predictions, actual)
        self.assertTrue((weights >= 0).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[0], weights[1])

    def test_streamlit_analysis_path_fetches_online_data_directly(self):
        data = sample_data(650)
        with patch("local_analysis.get_stock_data", return_value=data) as fetch, \
             patch("local_analysis.record_experiment", return_value="run-local"):
            result = analyze_symbol("AAPL", "daily", 1, "ridge")
        fetch.assert_called_once_with("AAPL", "daily", force_refresh=True)
        self.assertEqual(result["data_source"], "Yahoo Finance through yfinance")
        self.assertEqual(result["experiment_id"], "run-local")
        self.assertIn("forecast", result)
        self.assertIn("chart_data", result)

    def test_forecast_metrics_and_horizons(self):
        data = sample_data(650)
        for horizon in (1, 3, 5, 10, 20, 30):
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
                self.assertEqual(len(result.validation_predictions), result.validation_points)
                self.assertEqual(result.validation_predictions[-1]["timestamp"], data.index[-1].isoformat())
                self.assertIn("actual", result.validation_predictions[0])
                self.assertIn("regime", result.diagnostics)
                self.assertIn("ewma_volatility_pct_per_bar", result.risk)
                self.assertGreaterEqual(result.probability_positive, 0)
                self.assertLessEqual(result.probability_positive, 100)
                self.assertIn("95", result.uncertainty["intervals"])
                self.assertIn("holdout_coverage_pct", result.uncertainty["intervals"]["90"])
                self.assertIn("training_seconds", result.metrics)

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
        self.assertIn(result.model, ("Weighted Ensemble", "Naïve: unchanged close"))
        self.assertEqual(len(result.ensemble_weights), 5)

    def test_empty_or_short_history_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "250"):
            forecast(sample_data(120))
        with self.assertRaises(ValueError):
            forecast(sample_data(), horizon=2)


if __name__ == "__main__":
    unittest.main()

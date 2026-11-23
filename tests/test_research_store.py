import tempfile
import unittest
from pathlib import Path

from research_store import recent_experiments, record_experiment


class ResearchStoreTests(unittest.TestCase):
    def test_records_reproducible_run_and_filters_history(self):
        result = {
            "forecast": {
                "horizon": 3,
                "market_profile": "india",
                "model": "Ridge",
                "reference_price": 100.0,
                "forecast": 101.0,
                "return_forecast_pct": 1.0,
                "training_seconds": 0.25,
                "validation_end": "2026-01-01T00:00:00+00:00",
                "validation_points": 20,
                "metrics": {"mae": 1.2},
                "naive_metrics": {"mae": 1.5},
                "model_comparison": {"Ridge": {"mae": 1.2}},
                "uncertainty": {"method": "selection residuals"},
                "validation_predictions": [{"timestamp": "2026-01-01", "actual": 101.0, "predicted": 100.5, "unchanged_close": 100.0}],
                "ensemble_weights": {},
                "diagnostics": {"regime": "sideways"},
                "risk": {"ewma_volatility_pct_per_bar": 0.5},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "runs.db"
            run_id = record_experiment("RELIANCE.NS", "daily", "auto", result, ["close", "rsi"], "2024-01-01", database, external_data={"available": True, "provider": "Frankfurter"})
            runs = recent_experiments("RELIANCE.NS", db_path=database)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["experiment_id"], run_id)
            self.assertEqual(runs[0]["config"]["random_seed"], 17)
            self.assertEqual(runs[0]["config"]["features"], ["close", "rsi"])
            self.assertEqual(runs[0]["config"]["external_data"]["provider"], "Frankfurter")
            self.assertEqual(runs[0]["metrics"]["holdout_predictions"][0]["actual"], 101.0)
            self.assertEqual(recent_experiments("AAPL", db_path=database), [])


if __name__ == "__main__":
    unittest.main()

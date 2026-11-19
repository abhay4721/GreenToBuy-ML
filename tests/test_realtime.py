import json
import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from paper_trading import place_paper_order, positions
from realtime import RealtimeUnavailable, TwelveDataStream, provider_instrument
from server import app, realtime_feed, realtime_quote


class RealtimeFeedTests(unittest.TestCase):
    def test_provider_symbol_mapping(self):
        self.assertEqual(provider_instrument("RELIANCE.NS"), ("RELIANCE", "NSE", None))
        self.assertEqual(provider_instrument("EURUSD"), ("EUR/USD", None, "Forex"))
        self.assertEqual(provider_instrument("^NSEI"), ("NIFTY 50", "NSE", None))

    def test_stream_requires_api_key(self):
        with patch.dict(os.environ, {"TWELVE_DATA_API_KEY": ""}):
            stream = TwelveDataStream()
            with self.assertRaisesRegex(RealtimeUnavailable, "TWELVE_DATA_API_KEY"):
                asyncio.run(stream.subscribe("EURUSD"))

    def test_api_routes_and_missing_key_response(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/api/realtime/quote/{symbol}", paths)
        self.assertIn("/api/paper/orders", paths)
        with patch.object(realtime_feed, "api_key", None):
            with self.assertRaises(HTTPException) as result:
                asyncio.run(realtime_quote("EURUSD"))
        self.assertEqual(result.exception.status_code, 503)

    def test_price_event_is_normalized_to_ticker(self):
        stream = TwelveDataStream(api_key="test")
        stream._subscriptions["EURUSD"] = {"ticker": "EURUSD", "symbol": "EUR/USD", "exchange": None, "type": "Forex"}
        stream._handle_message(json.dumps({
            "event": "price", "symbol": "EUR/USD", "exchange": "FOREX", "timestamp": int(time.time()),
            "price": 1.08, "bid": 1.0799, "ask": 1.0801, "currency_quote": "USD",
        }))
        quote = stream.quote("EURUSD")
        self.assertEqual(quote["price"], 1.08)
        self.assertEqual(quote["ask"], 1.0801)
        self.assertEqual(quote["currency"], "USD")


class PaperTradingTests(unittest.TestCase):
    def test_buy_sell_positions_and_realized_profit(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("paper_trading.PAPER_DB_PATH", Path(temp) / "paper.db"):
                quote = {"status": "live", "symbol": "EURUSD", "price": 1.10, "bid": 1.099, "ask": 1.101,
                         "currency": "USD", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
                buy = place_paper_order("EURUSD", "BUY", 100, quote)
                quote["price"], quote["bid"], quote["ask"] = 1.20, 1.199, 1.201
                sell = place_paper_order("EURUSD", "SELL", 40, quote)
                self.assertEqual(buy["fill_price"], 1.101)
                self.assertEqual(sell["fill_price"], 1.199)
                position = positions()[0]
                self.assertEqual(position["quantity"], 60)
                self.assertAlmostEqual(position["realized_pnl"], (1.199 - 1.101) * 40)

    def test_rejects_oversell_and_stale_quote(self):
        quote = {"status": "live", "symbol": "AAPL", "price": 200, "timestamp": "2020-01-01T00:00:00+00:00"}
        with tempfile.TemporaryDirectory() as temp:
            with patch("paper_trading.PAPER_DB_PATH", Path(temp) / "paper.db"):
                with self.assertRaisesRegex(ValueError, "60 seconds old"):
                    place_paper_order("AAPL", "BUY", 1, quote)
                quote["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
                with self.assertRaisesRegex(ValueError, "paper position"):
                    place_paper_order("AAPL", "SELL", 1, quote)

    def test_indian_equity_paper_quantity_must_be_whole_shares(self):
        quote = {"status": "live", "symbol": "RELIANCE.NS", "price": 1500,
                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
        with self.assertRaisesRegex(ValueError, "whole shares"):
            place_paper_order("RELIANCE.NS", "BUY", 1.5, quote)


if __name__ == "__main__":
    unittest.main()

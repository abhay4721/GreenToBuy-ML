"""Shared Twelve Data WebSocket feed for real-time price snapshots."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from urllib.parse import quote

from websockets.asyncio.client import connect

from market_data import normalize_symbol

logger = logging.getLogger(__name__)
FOREX_SYMBOLS = {"EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "XAUUSD": "XAU/USD"}
INDEX_SYMBOLS = {"^NSEI": ("NIFTY 50", "NSE"), "^NSEBANK": ("NIFTY BANK", "NSE"), "^BSESN": ("SENSEX", "BSE")}
try:
    MAX_REALTIME_SYMBOLS = max(1, int(os.getenv("MAX_REALTIME_SYMBOLS", "8")))
except ValueError:
    MAX_REALTIME_SYMBOLS = 8


class RealtimeUnavailable(RuntimeError):
    """Raised when realtime quotes are not configured or available."""


def provider_instrument(symbol: str) -> tuple[str, str | None, str | None]:
    ticker = normalize_symbol(symbol)
    if ticker in FOREX_SYMBOLS:
        return FOREX_SYMBOLS[ticker], None, "Forex"
    if ticker in INDEX_SYMBOLS:
        name, exchange = INDEX_SYMBOLS[ticker]
        return name, exchange, None
    if ticker.endswith(".NS"):
        return ticker[:-3], "NSE", None
    if ticker.endswith(".BO"):
        return ticker[:-3], "BSE", None
    return ticker, None, None


class TwelveDataStream:
    """Maintain one upstream socket and fan out its latest prices by symbol."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TWELVE_DATA_API_KEY")
        self._socket = None
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._subscriptions: dict[str, dict[str, str | None]] = {}
        self._latest: dict[str, dict] = {}
        self._errors: dict[str, str] = {}
        self.connected = False
        self.last_error: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def subscribe(self, symbol: str) -> dict | None:
        ticker = normalize_symbol(symbol)
        if not self.configured:
            raise RealtimeUnavailable("Set TWELVE_DATA_API_KEY to enable live quotes.")
        instrument, exchange, asset_type = provider_instrument(ticker)
        subscription = {"ticker": ticker, "symbol": instrument, "exchange": exchange, "type": asset_type}
        async with self._lock:
            is_new = ticker not in self._subscriptions
            if is_new and len(self._subscriptions) >= MAX_REALTIME_SYMBOLS:
                raise RealtimeUnavailable(f"This process has reached its {MAX_REALTIME_SYMBOLS}-symbol stream limit. Unsubscribe an old symbol or increase MAX_REALTIME_SYMBOLS to match your data plan.")
            self._subscriptions[ticker] = subscription
            socket = self._socket
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._run(), name="twelve-data-stream")
        if is_new and socket is not None:
            await self._send_subscribe(socket, subscription)
        return self._latest.get(ticker)

    async def unsubscribe(self, symbol: str) -> None:
        ticker = normalize_symbol(symbol)
        async with self._lock:
            subscription = self._subscriptions.pop(ticker, None)
            self._latest.pop(ticker, None)
            self._errors.pop(ticker, None)
            socket = self._socket
        if subscription and socket is not None:
            payload = {key: value for key, value in subscription.items() if key != "ticker" and value}
            await socket.send(json.dumps({"action": "unsubscribe", "params": {"symbols": [payload]}}))

    def quote(self, symbol: str) -> dict | None:
        return self._latest.get(normalize_symbol(symbol))

    def error(self, symbol: str) -> str | None:
        return self._errors.get(normalize_symbol(symbol)) or self.last_error

    def status(self) -> dict:
        return {
            "provider": "Twelve Data",
            "configured": self.configured,
            "connected": self.connected,
            "subscriptions": sorted(self._subscriptions),
            "last_error": self.last_error,
        }

    async def close(self) -> None:
        async with self._lock:
            socket, task = self._socket, self._task
            self._socket = None
            self._task = None
        if socket is not None:
            await socket.close()
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _send_subscribe(self, socket, subscription: dict) -> None:
        payload = {key: value for key, value in subscription.items() if key != "ticker" and value}
        await socket.send(json.dumps({"action": "subscribe", "params": {"symbols": [payload]}}))

    def _handle_message(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return
        event = message.get("event")
        if event == "subscribe-status":
            for failed in message.get("fails", []):
                source_symbol = str(failed.get("symbol", "")).upper()
                for ticker, subscription in self._subscriptions.items():
                    if subscription["symbol"].upper() == source_symbol:
                        self._errors[ticker] = str(failed.get("message") or failed.get("code") or "Subscription rejected by provider.")
            return
        if event != "price":
            return
        source_symbol = str(message.get("symbol", "")).upper()
        source_exchange = str(message.get("exchange", "")).upper()
        ticker = next((
            key for key, item in self._subscriptions.items()
            if item["symbol"].upper() == source_symbol and (not item["exchange"] or item["exchange"].upper() == source_exchange)
        ), None)
        if ticker is None:
            ticker = next((key for key, item in self._subscriptions.items() if item["symbol"].upper() == source_symbol), None)
        if ticker is None:
            return
        try:
            timestamp = datetime.fromtimestamp(float(message["timestamp"]), tz=timezone.utc).isoformat()
            price = float(message["price"])
        except (KeyError, TypeError, ValueError, OverflowError):
            self._errors[ticker] = "Provider sent a malformed price update."
            return
        quote = {
            "symbol": ticker,
            "provider_symbol": message.get("symbol"),
            "exchange": message.get("exchange"),
            "price": price,
            "bid": float(message["bid"]) if message.get("bid") not in (None, "") else None,
            "ask": float(message["ask"]) if message.get("ask") not in (None, "") else None,
            "currency": message.get("currency_quote") or message.get("currency") or ("INR" if ticker.endswith((".NS", ".BO")) else None),
            "timestamp": timestamp,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "day_volume": message.get("day_volume"),
            "provider": "Twelve Data",
        }
        self._latest[ticker] = quote
        self._errors.pop(ticker, None)

    async def _run(self) -> None:
        url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={quote(self.api_key or '')}"
        while True:
            try:
                async with connect(url, ping_interval=None, close_timeout=3) as socket:
                    async with self._lock:
                        self._socket = socket
                        subscriptions = list(self._subscriptions.values())
                    self.connected = True
                    self.last_error = None
                    if subscriptions:
                        symbols = [
                            {key: value for key, value in item.items() if key != "ticker" and value}
                            for item in subscriptions
                        ]
                        await socket.send(json.dumps({"action": "subscribe", "params": {"symbols": symbols}}))
                    while True:
                        try:
                            raw = await asyncio.wait_for(socket.recv(), timeout=10)
                            self._handle_message(raw)
                        except asyncio.TimeoutError:
                            await socket.send(json.dumps({"action": "heartbeat"}))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.info("Twelve Data stream disconnected (%s). Retrying shortly.", type(exc).__name__)
                self.connected = False
                self.last_error = "Unable to connect to the quote provider; check the API key, network, and symbol entitlement."
                async with self._lock:
                    self._socket = None
                await asyncio.sleep(3)

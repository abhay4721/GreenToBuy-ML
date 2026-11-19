"""Yahoo Finance acquisition, validation, retry, and local caching."""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent
DB_PATH = DATA_DIR / "stock_data.db"
INTERVALS = {
    "5min": ("7d", "5m", timedelta(minutes=15)),
    "15min": ("60d", "15m", timedelta(minutes=30)),
    "60min": ("730d", "60m", timedelta(hours=2)),
    "daily": ("5y", "1d", timedelta(hours=12)),
}
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
YAHOO_TICKERS = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X"}
INDIAN_INDEX_TICKERS = {"^NSEI", "^NSEBANK", "^BSESN"}
FOREX_TICKERS = {"XAUUSD", "EURUSD", "GBPUSD"}


def normalize_symbol(symbol: str) -> str:
    value = symbol.strip().upper()
    if not value or len(value) > 20 or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^=" for c in value):
        raise ValueError("Enter a valid ticker symbol (letters, numbers, dots, hyphens, ^, or =).")
    return value


def yahoo_ticker(symbol: str) -> str:
    """Resolve the legacy precious metal and forex aliases supported by the UI."""
    ticker = normalize_symbol(symbol)
    return YAHOO_TICKERS.get(ticker, ticker)


def market_profile(symbol: str) -> str:
    """Classify supported symbols for market-specific model defaults."""
    ticker = normalize_symbol(symbol)
    if ticker in FOREX_TICKERS:
        return "forex"
    if ticker.endswith((".NS", ".BO")) or ticker in INDIAN_INDEX_TICKERS:
        return "india"
    return "global_equity"


def validate_market_data(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        raise ValueError("No market data was returned for this symbol and interval.")
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = [str(col[0]).lower().replace(" ", "_") for col in data.columns]
    else:
        data = data.copy()
        data.columns = [str(col).lower().replace(" ", "_") for col in data.columns]
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Market data is missing required fields: {', '.join(missing)}")
    data = data[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data.index = pd.to_datetime(data.index, utc=True)
    data.index.name = "date"
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data = data.dropna(subset=["open", "high", "low", "close"])
    data = data[(data[["open", "high", "low", "close"]] > 0).all(axis=1)]
    if data.empty:
        raise ValueError("Market data contained no valid price rows.")
    return data


def _read_cache(symbol: str, interval: str) -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cached = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM market_data WHERE symbol=? AND interval=? ORDER BY date",
                conn, params=(symbol, interval), parse_dates=["date"],
            )
        if cached.empty:
            return pd.DataFrame()
        cached = cached.set_index("date")
        cached.index = pd.to_datetime(cached.index, utc=True)
        return cached
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()


def _write_cache(symbol: str, interval: str, data: pd.DataFrame) -> None:
    rows = data.reset_index()
    values = [
        (symbol, interval, pd.Timestamp(row.date).isoformat(), float(row.open), float(row.high), float(row.low), float(row.close), float(row.volume))
        for row in rows.itertuples(index=False)
    ]
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS market_data (
            symbol TEXT NOT NULL, interval TEXT NOT NULL, date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY(symbol, interval, date))""")
        conn.executemany(
            "INSERT OR REPLACE INTO market_data(symbol, interval, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        conn.commit()


def get_stock_data(symbol: str, interval: str = "daily") -> pd.DataFrame:
    symbol = normalize_symbol(symbol)
    if interval not in INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")
    period, yf_interval, max_age = INTERVALS[interval]
    cached = _read_cache(symbol, interval)
    now = datetime.now(timezone.utc)
    if not cached.empty and now - cached.index[-1].to_pydatetime() < max_age:
        return cached

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            import yfinance as yf

            frame = yf.download(yahoo_ticker(symbol), period=period, interval=yf_interval, progress=False, auto_adjust=False, threads=False)
            validated = validate_market_data(frame)
            try:
                _write_cache(symbol, interval, validated)
            except sqlite3.Error as exc:
                logger.warning("Could not cache %s %s: %s", symbol, interval, exc)
            return validated
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))

    if not cached.empty:
        logger.warning("Using stale cache after fetch failure for %s %s: %s", symbol, interval, last_error)
        return cached
    raise RuntimeError(f"Could not fetch {symbol} market data: {last_error}") from last_error

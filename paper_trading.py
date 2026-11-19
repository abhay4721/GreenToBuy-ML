"""Local-only simulated orders and positions; never calls a broker."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from market_data import market_profile, normalize_symbol

PAPER_DB_PATH = Path(__file__).resolve().parent / "paper_trading.db"


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(PAPER_DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS paper_orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
        quantity REAL NOT NULL CHECK(quantity > 0),
        fill_price REAL NOT NULL CHECK(fill_price > 0),
        currency TEXT,
        quote_timestamp TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    return connection


def positions() -> list[dict]:
    with _connect() as connection:
        orders = connection.execute("SELECT * FROM paper_orders ORDER BY id").fetchall()
    state: dict[str, dict] = {}
    for order in orders:
        item = state.setdefault(order["symbol"], {
            "symbol": order["symbol"], "quantity": 0.0, "average_price": 0.0,
            "realized_pnl": 0.0, "currency": order["currency"],
        })
        quantity, price = float(order["quantity"]), float(order["fill_price"])
        if order["side"] == "BUY":
            total = item["quantity"] + quantity
            item["average_price"] = (item["average_price"] * item["quantity"] + price * quantity) / total
            item["quantity"] = total
        else:
            item["realized_pnl"] += (price - item["average_price"]) * quantity
            item["quantity"] -= quantity
            if item["quantity"] <= 1e-10:
                item["quantity"] = 0.0
                item["average_price"] = 0.0
    return [item for item in state.values() if item["quantity"] > 0]


def place_paper_order(symbol: str, side: str, quantity: float, quote: dict) -> dict:
    ticker = normalize_symbol(symbol)
    side = side.strip().upper()
    quantity = float(quantity)
    if side not in ("BUY", "SELL"):
        raise ValueError("Side must be BUY or SELL.")
    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")
    if market_profile(ticker) == "india" and ticker.startswith("^"):
        raise ValueError("Index symbols are not tradeable instruments in paper mode.")
    if market_profile(ticker) == "india" and not quantity.is_integer():
        raise ValueError("Indian equity paper-order quantities must be whole shares.")
    if quote.get("status") == "connecting" or quote.get("price") is None:
        raise ValueError("Wait for a live quote before placing a paper order.")
    if quote.get("symbol") != ticker:
        raise ValueError("Live quote symbol does not match the order symbol.")
    try:
        quote_time = datetime.fromisoformat(str(quote["timestamp"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as exc:
        raise ValueError("Live quote has no valid source timestamp.") from exc
    if quote_time.tzinfo is None:
        quote_time = quote_time.replace(tzinfo=timezone.utc)
    if (datetime.now(timezone.utc) - quote_time).total_seconds() > 60:
        raise ValueError("Live quote is over 60 seconds old; wait for a fresh update.")

    side_price = quote.get("ask") if side == "BUY" else quote.get("bid")
    fill_price = float(side_price if side_price is not None else quote["price"])
    if fill_price <= 0:
        raise ValueError("Quote price must be positive.")
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        if side == "SELL":
            rows = connection.execute("SELECT side, quantity FROM paper_orders WHERE symbol=? ORDER BY id", (ticker,)).fetchall()
            held = sum((1 if row["side"] == "BUY" else -1) * float(row["quantity"]) for row in rows)
            if quantity > held + 1e-10:
                raise ValueError(f"Cannot paper-sell {quantity:g}; current paper position is {held:g}.")
        cursor = connection.execute(
            "INSERT INTO paper_orders(symbol, side, quantity, fill_price, currency, quote_timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ticker, side, quantity, fill_price, quote.get("currency"), str(quote["timestamp"]), datetime.now(timezone.utc).isoformat()),
        )
        order_id = cursor.lastrowid
    return {
        "mode": "paper",
        "order_id": order_id,
        "symbol": ticker,
        "side": side,
        "quantity": quantity,
        "fill_price": fill_price,
        "currency": quote.get("currency"),
        "quote_timestamp": quote["timestamp"],
        "message": "Simulated locally. No broker order was sent.",
    }


def order_history(limit: int = 100) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute("SELECT * FROM paper_orders ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]

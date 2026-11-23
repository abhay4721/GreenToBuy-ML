"""Direct in-process data acquisition and forecast pipeline for Streamlit."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from forecasting import build_features, forecast
from external_data import indian_macro_features
from market_data import get_stock_data, market_profile, normalize_symbol
from research_store import record_experiment

logger = logging.getLogger(__name__)


def analyze_symbol(symbol: str, interval: str, horizon: int, model: str) -> dict:
    """Fetch current history, evaluate a forecast, and save its research run."""
    ticker = normalize_symbol(symbol)
    data = get_stock_data(ticker, interval, force_refresh=True)
    profile = market_profile(ticker)
    external_features, external_status = indian_macro_features(data.index) if profile == "india" else (None, {"available": False, "reason": "No additional macro feed is configured for this market."})
    result = forecast(data, horizon=horizon, model_choice=model, market_profile=profile, external_features=external_features)
    features = build_features(data, external_features)
    latest = features.iloc[-1]
    indicator_names = ("sma_20", "sma_50", "sma_200", "rsi", "macd", "atr", "adx", "bb_upper", "bb_lower")
    indicators = {name: (float(latest[name]) if pd.notna(latest[name]) else None) for name in indicator_names}
    rows = data.tail(250 if interval == "daily" else 300)
    response = {
        "symbol": ticker,
        "market_profile": result.market_profile,
        "interval": interval,
        "horizon": horizon,
        "data_source": "Yahoo Finance through yfinance",
        "external_data": external_status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "latest_price": {"price": result.reference_price, "timestamp": data.index[-1].isoformat()},
        "forecast": result.to_dict(),
        "diagnostics": result.diagnostics,
        "risk": result.risk,
        "indicators": indicators,
        "chart_data": [
            {"x": timestamp.isoformat(), "o": float(row.open), "h": float(row.high), "l": float(row.low), "c": float(row.close), "v": float(row.volume) if pd.notna(row.volume) else None}
            for timestamp, row in rows.iterrows()
        ],
    }
    try:
        response["experiment_id"] = record_experiment(
            ticker,
            interval,
            model,
            response,
            list(features.columns),
            str(features.dropna().index[0].isoformat()),
            external_data=external_status,
        )
    except Exception:
        logger.exception("Could not persist research run for %s", ticker)
        response["experiment_id"] = None
    return response

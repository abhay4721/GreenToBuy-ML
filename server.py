"""HTTP API for market data and validated forecasts."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from forecasting import build_features, forecast
from market_data import INTERVALS, get_stock_data, normalize_symbol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="GreenToBuy Market Analytics", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/stock/{symbol}")
def get_stock_analysis(
    symbol: str,
    interval: str = Query("daily", description="One of 5min, 15min, 60min, daily"),
    horizon: int = Query(1, description="Trading bars ahead: 1, 5, 10, or 30"),
    model: str = Query("auto", description="auto, weighted ensemble, ridge, random forest, extra trees, hist gradient boosting, or svr"),
) -> dict:
    try:
        ticker = normalize_symbol(symbol)
        if interval not in INTERVALS:
            raise HTTPException(status_code=422, detail=f"Interval must be one of: {', '.join(INTERVALS)}")
        if horizon not in (1, 5, 10, 30):
            raise HTTPException(status_code=422, detail="Horizon must be 1, 5, 10, or 30 trading bars.")
        if model.lower() not in ("auto", "weighted ensemble", "ridge", "random forest", "extra trees", "hist gradient boosting", "svr"):
            raise HTTPException(status_code=422, detail="Choose auto, weighted ensemble, ridge, random forest, extra trees, hist gradient boosting, or svr.")
        data = get_stock_data(ticker, interval)
        result = forecast(data, horizon=horizon, model_choice=model)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Analysis failed for %s", symbol)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    features = build_features(data)
    latest_features = features.iloc[-1]
    indicator_names = ("sma_20", "sma_50", "sma_200", "rsi", "macd", "atr", "adx", "bb_upper", "bb_lower")
    indicators = {name: (float(latest_features[name]) if pd.notna(latest_features[name]) else None) for name in indicator_names}
    rows = data.tail(250 if interval == "daily" else 300)
    return {
        "symbol": ticker,
        "interval": interval,
        "horizon": horizon,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "latest_price": {"price": result.reference_price, "timestamp": data.index[-1].isoformat()},
        "forecast": result.to_dict(),
        "indicators": indicators,
        "chart_data": [
            {"x": timestamp.isoformat(), "o": float(row.open), "h": float(row.high), "l": float(row.low), "c": float(row.close), "v": float(row.volume) if pd.notna(row.volume) else None}
            for timestamp, row in rows.iterrows()
        ],
    }

"""HTTP API for market data and validated forecasts."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from forecasting import build_features, forecast
from market_data import INTERVALS, get_stock_data, market_profile, normalize_symbol
from paper_trading import order_history, place_paper_order, positions
from realtime import RealtimeUnavailable, TwelveDataStream
from research_store import recent_experiments, record_experiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="GreenToBuy Market Analytics", version="2.0.0")
realtime_feed = TwelveDataStream()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


class PaperOrderRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    side: Literal["BUY", "SELL"]
    quantity: float = Field(gt=0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/research/history")
def research_history(
    symbol: str | None = Query(None, min_length=1, max_length=20),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict]:
    try:
        normalized = normalize_symbol(symbol) if symbol else None
        return recent_experiments(normalized, limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/realtime/status")
def realtime_status() -> dict:
    return realtime_feed.status()


@app.get("/api/realtime/quote/{symbol}")
async def realtime_quote(symbol: str) -> dict:
    try:
        ticker = normalize_symbol(symbol)
        latest = await realtime_feed.subscribe(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RealtimeUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if latest is not None:
        tick_time = datetime.fromisoformat(latest["timestamp"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - tick_time).total_seconds()
        return {"status": "live" if age <= 60 else "stale", "age_seconds": max(0, age), **latest}
    provider_error = realtime_feed.error(ticker)
    if provider_error:
        return {"status": "error", "symbol": ticker, "provider": "Twelve Data", "detail": provider_error}
    return {"status": "connecting", "symbol": ticker, "provider": "Twelve Data"}


@app.delete("/api/realtime/quote/{symbol}")
async def stop_realtime_quote(symbol: str) -> dict[str, str]:
    try:
        await realtime_feed.unsubscribe(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "unsubscribed", "symbol": normalize_symbol(symbol)}


@app.post("/api/paper/orders")
def submit_paper_order(request: PaperOrderRequest) -> dict:
    try:
        ticker = normalize_symbol(request.symbol)
        quote = realtime_feed.quote(ticker)
        if quote is None:
            raise HTTPException(status_code=409, detail="Subscribe to a live quote and wait for a fresh tick before ordering.")
        return place_paper_order(ticker, request.side, request.quantity, quote)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/paper/positions")
def get_paper_positions() -> list[dict]:
    return positions()


@app.get("/api/paper/orders")
def get_paper_orders(limit: int = Query(100, ge=1, le=500)) -> list[dict]:
    return order_history(limit)


@app.on_event("shutdown")
async def shutdown_realtime_feed() -> None:
    await realtime_feed.close()


@app.get("/api/stock/{symbol}")
def get_stock_analysis(
    symbol: str,
    interval: str = Query("daily", description="One of 5min, 15min, 60min, daily"),
    horizon: int = Query(1, description="Trading bars ahead: 1, 3, 5, 10, 20, or 30"),
    model: str = Query("auto", description="auto, weighted ensemble, ridge, random forest, extra trees, hist gradient boosting, or svr"),
) -> dict:
    try:
        ticker = normalize_symbol(symbol)
        if interval not in INTERVALS:
            raise HTTPException(status_code=422, detail=f"Interval must be one of: {', '.join(INTERVALS)}")
        if horizon not in (1, 3, 5, 10, 20, 30):
            raise HTTPException(status_code=422, detail="Horizon must be 1, 3, 5, 10, 20, or 30 trading bars.")
        if model.lower() not in ("auto", "weighted ensemble", "ridge", "random forest", "extra trees", "hist gradient boosting", "svr"):
            raise HTTPException(status_code=422, detail="Choose auto, weighted ensemble, ridge, random forest, extra trees, hist gradient boosting, or svr.")
        data = get_stock_data(ticker, interval)
        profile = market_profile(ticker)
        result = forecast(data, horizon=horizon, model_choice=model, market_profile=profile)
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
    response = {
        "symbol": ticker,
        "market_profile": result.market_profile,
        "interval": interval,
        "horizon": horizon,
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
        )
    except Exception:
        logger.exception("Could not persist research run for %s", ticker)
        response["experiment_id"] = None
    return response

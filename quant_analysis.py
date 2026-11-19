"""Causal return diagnostics, rule-based regime labels, and volatility estimates."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _finite(value: float | None) -> float | None:
    """Convert finite scalar diagnostics to JSON-safe values."""
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _hurst_exponent(values: pd.Series) -> float | None:
    sample = values.dropna().tail(500).to_numpy(dtype=float)
    lags = np.array([2, 4, 8, 16, 32])
    if len(sample) < int(lags[-1]) * 4:
        return None
    scales = np.array([np.std(sample[lag:] - sample[:-lag]) for lag in lags])
    valid = scales > 0
    if valid.sum() < 3:
        return None
    slope = np.polyfit(np.log(lags[valid]), np.log(scales[valid]), 1)[0]
    return _finite(float(np.clip(slope, 0.0, 1.0)))


def analyze_market(data: pd.DataFrame, horizon: int) -> tuple[dict, dict]:
    """Summarize only historical bars up to the latest observation."""
    close = data["close"].astype(float)
    open_price = data["open"].astype(float)
    high, low = data["high"].astype(float), data["low"].astype(float)
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sample = returns.tail(252)
    rolling_vol = returns.rolling(20, min_periods=20).std().dropna()
    current_vol = float(rolling_vol.iloc[-1]) if len(rolling_vol) else 0.0
    vol_reference = float(rolling_vol.tail(252).median()) if len(rolling_vol) else 0.0
    trend_return = float(close.iloc[-1] / close.iloc[max(0, len(close) - 21)] - 1)
    sma50 = float(close.rolling(50, min_periods=1).mean().iloc[-1])
    if trend_return >= 0.02 and close.iloc[-1] > sma50:
        trend_regime = "bullish trend"
    elif trend_return <= -0.02 and close.iloc[-1] < sma50:
        trend_regime = "bearish trend"
    else:
        trend_regime = "sideways or mixed"
    vol_regime = "high volatility" if vol_reference and current_vol > vol_reference else "normal or low volatility"

    log_hl = np.log(high / low).replace([np.inf, -np.inf], np.nan)
    log_co = np.log(close / open_price).replace([np.inf, -np.inf], np.nan)
    parkinson_var = (log_hl.pow(2) / (4 * math.log(2))).tail(20).mean()
    garman_klass_var = (0.5 * log_hl.pow(2) - (2 * math.log(2) - 1) * log_co.pow(2)).tail(20).mean()
    ewma_variance = 0.0
    for value in returns.tail(500).to_numpy(dtype=float):
        ewma_variance = 0.94 * ewma_variance + 0.06 * value * value
    ewma_vol = math.sqrt(max(ewma_variance, 0.0))
    horizon_vol = ewma_vol * math.sqrt(horizon)
    reference_price = float(close.iloc[-1])
    wealth = (1 + sample).cumprod()
    drawdown = wealth / wealth.cummax() - 1 if len(wealth) else pd.Series(dtype=float)

    diagnostics = {
        "observation_count": int(len(data)),
        "return_observation_count": int(len(sample)),
        "mean_return_pct_per_bar": _finite(sample.mean() * 100) if len(sample) else 0.0,
        "median_return_pct_per_bar": _finite(sample.median() * 100) if len(sample) else 0.0,
        "return_std_pct_per_bar": _finite(sample.std() * 100) if len(sample) > 1 else 0.0,
        "return_skew": _finite(sample.skew()) if len(sample) > 2 else 0.0,
        "return_excess_kurtosis": _finite(sample.kurt()) if len(sample) > 3 else 0.0,
        "lag1_return_autocorrelation": _finite(sample.autocorr(1)) if len(sample) > 2 else 0.0,
        "hurst_estimate": _hurst_exponent(sample),
        "trailing_20_bar_return_pct": trend_return * 100,
        "regime": f"{trend_regime}; {vol_regime}",
        "regime_method": "Trailing 20-bar return, 50-bar trend, and trailing volatility median; rule-based, no future bars.",
        "max_drawdown_last_252_bars_pct": _finite(drawdown.min() * 100) if len(drawdown) else 0.0,
    }
    risk = {
        "ewma_volatility_pct_per_bar": ewma_vol * 100,
        "sqrt_time_horizon_volatility_pct": horizon_vol * 100,
        "parkinson_volatility_pct_per_bar": math.sqrt(max(float(parkinson_var), 0.0)) * 100 if pd.notna(parkinson_var) else 0.0,
        "garman_klass_volatility_pct_per_bar": math.sqrt(max(float(garman_klass_var), 0.0)) * 100 if pd.notna(garman_klass_var) else 0.0,
        "sqrt_time_range_lower": max(0.0, reference_price * (1 - 1.96 * horizon_vol)),
        "sqrt_time_range_upper": reference_price * (1 + 1.96 * horizon_vol),
        "range_assumption": "EWMA volatility scaled by square-root-of-time; descriptive risk range, not a calibrated prediction interval.",
    }
    return diagnostics, risk

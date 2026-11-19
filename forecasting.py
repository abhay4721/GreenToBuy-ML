"""Leakage-conscious features and chronological stock forecast evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ForecastResult:
    forecast: float
    reference_price: float
    horizon: int
    model: str
    metrics: dict[str, float]
    naive_metrics: dict[str, float]
    uncertainty: dict[str, float]
    validation_points: int
    validation_end: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build features from current and prior bars only; never use future values."""
    close = data["close"].astype(float)
    high, low = data["high"].astype(float), data["low"].astype(float)
    volume = data["volume"].astype(float).fillna(0)
    f = pd.DataFrame(index=data.index)
    f["close"] = close
    for window in (20, 50, 100, 200):
        f[f"sma_{window}"] = close.rolling(window, min_periods=window).mean()
    for span in (12, 26, 50):
        f[f"ema_{span}"] = close.ewm(span=span, min_periods=span, adjust=False).mean()
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    rs = gain.rolling(14, min_periods=14).mean() / loss.rolling(14, min_periods=14).mean().replace(0, np.nan)
    f["rsi"] = 100 - 100 / (1 + rs)
    f["macd"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    f["macd_signal"] = f["macd"].ewm(span=9, adjust=False).mean()
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std()
    f["bb_upper"], f["bb_lower"] = mid + 2 * std, mid - 2 * std
    previous = close.shift(1)
    true_range = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
    f["atr"] = true_range.rolling(14, min_periods=14).mean()
    plus_dm = high.diff().where((high.diff() > -low.diff()) & (high.diff() > 0), 0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0)
    plus_di = 100 * plus_dm.rolling(14, min_periods=14).sum() / f["atr"].replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14, min_periods=14).sum() / f["atr"].replace(0, np.nan)
    f["adx"] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).rolling(14, min_periods=14).mean()
    direction = np.sign(close.diff()).fillna(0)
    f["obv"] = (direction * volume).cumsum()
    f["daily_return"] = close.pct_change()
    f["log_return"] = np.log(close / previous)
    f["volatility_20"] = f["daily_return"].rolling(20, min_periods=20).std()
    f["momentum_10"] = close / close.shift(10) - 1
    f["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    f["high_low_range"] = (high - low) / close
    for lag in (1, 2, 3, 5, 10):
        f[f"close_lag_{lag}"] = close.shift(lag)
        f[f"return_lag_{lag}"] = f["daily_return"].shift(lag)
    for window in (5, 10, 20):
        f[f"rolling_mean_return_{window}"] = f["daily_return"].rolling(window).mean()
        f[f"rolling_std_return_{window}"] = f["daily_return"].rolling(window).std()
    return f.replace([np.inf, -np.inf], np.nan)


def _metrics(actual: np.ndarray, predicted: np.ndarray, previous: np.ndarray) -> dict[str, float]:
    nonzero = np.abs(actual) > 1e-12
    direction = (np.sign(actual - previous) == np.sign(predicted - previous)).mean()
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else 0.0,
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else 0.0,
        "directional_accuracy": float(direction * 100),
        "mean_error_pct": float(np.mean(np.abs(actual - predicted) / np.maximum(np.abs(actual), 1e-12)) * 100),
    }


def _model(name: str):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    return RandomForestRegressor(n_estimators=160, min_samples_leaf=4, max_features=0.8, random_state=17, n_jobs=1)


def forecast(data: pd.DataFrame, horizon: int = 1, model_choice: str = "auto") -> ForecastResult:
    if horizon not in (1, 5, 10, 30):
        raise ValueError("Horizon must be 1, 5, 10, or 30 trading days.")
    if len(data) < 250:
        raise ValueError("At least 250 valid price rows are required to calculate the 200-bar trend features and evaluate a forecast.")
    features = build_features(data)
    # Direct horizon target: features at t predict close at t+horizon.
    target = data["close"].shift(-horizon).rename("target")
    dataset = features.join(target).replace([np.inf, -np.inf], np.nan).dropna()
    if len(dataset) < 35:
        raise ValueError("Insufficient complete feature rows for chronological evaluation.")
    x, y = dataset.drop(columns="target"), dataset["target"].to_numpy(dtype=float)
    prior_close = data["close"].reindex(dataset.index).to_numpy(dtype=float)
    # Keep evaluation windows bounded and purge horizon-overlapping labels.
    test_size = min(30, max(10, len(dataset) // 5))
    if len(dataset) < 40 + 2 * test_size + 2 * horizon:
        raise ValueError("Not enough history for two purged walk-forward folds at this forecast horizon.")
    splitter = TimeSeriesSplit(n_splits=2, test_size=test_size, gap=horizon)
    predictions: dict[str, list[float]] = {"Ridge": [], "Random Forest": []}
    actual: list[float] = []
    previous: list[float] = []
    for train_idx, test_idx in splitter.split(x):
        for name in predictions:
            model = _model("Ridge" if name == "Ridge" else "Random Forest")
            model.fit(x.iloc[train_idx], y[train_idx])
            predictions[name].extend(model.predict(x.iloc[test_idx]).tolist())
        actual.extend(y[test_idx].tolist())
        previous.extend(prior_close[test_idx].tolist())
    actual_array, previous_array = np.array(actual), np.array(previous)
    scores = {name: _metrics(actual_array, np.asarray(pred), previous_array) for name, pred in predictions.items()}
    naive = _metrics(actual_array, previous_array, previous_array)
    selected = min(scores, key=lambda name: scores[name]["mae"])
    if model_choice.lower() not in ("auto", "ridge", "random forest"):
        raise ValueError("Model must be auto, ridge, or random forest.")
    if model_choice.lower() != "auto":
        selected = "Ridge" if model_choice.lower() == "ridge" else "Random Forest"
    fitted = _model(selected)
    fitted.fit(x, y)
    latest_features = features.iloc[[-1]].replace([np.inf, -np.inf], np.nan)
    if latest_features.isna().any(axis=None):
        raise ValueError("Latest bar does not have enough history for all features.")
    value = float(fitted.predict(latest_features)[0])
    residuals = np.abs(actual_array - np.asarray(predictions[selected]))
    radius = float(np.quantile(residuals, 0.9))
    return ForecastResult(
        forecast=value,
        reference_price=float(data["close"].iloc[-1]),
        horizon=horizon,
        model=selected,
        metrics=scores[selected],
        naive_metrics=naive,
        uncertainty={"lower": max(0.0, value - radius), "upper": value + radius, "method": "90th percentile of walk-forward absolute errors"},
        validation_points=len(actual_array),
        validation_end=str(dataset.index[-horizon].isoformat() if hasattr(dataset.index[-horizon], "isoformat") else dataset.index[-horizon]),
    )

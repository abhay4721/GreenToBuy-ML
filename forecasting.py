"""Causal financial features and nested chronological model comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from quant_analysis import analyze_market

MAX_TRAINING_ROWS = 1_500
MIN_SELECTION_EDGE = 0.05


@dataclass
class ForecastResult:
    forecast: float
    reference_price: float
    horizon: int
    market_profile: str
    model: str
    metrics: dict[str, float]
    naive_metrics: dict[str, float]
    model_comparison: dict[str, dict[str, float]]
    ensemble_weights: dict[str, float]
    diagnostics: dict[str, Any]
    risk: dict[str, Any]
    uncertainty: dict[str, Any]
    training_seconds: float
    return_forecast_pct: float
    probability_positive: float
    validation_predictions: list[dict[str, Any]]
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
    f["overnight_gap"] = data["open"].astype(float) / previous - 1
    f["intraday_return"] = close / data["open"].astype(float) - 1
    candle_top = pd.concat([data["open"], close], axis=1).max(axis=1)
    candle_bottom = pd.concat([data["open"], close], axis=1).min(axis=1)
    f["candle_body"] = (close - data["open"].astype(float)) / close
    f["upper_wick"] = (high - candle_top) / close
    f["lower_wick"] = (candle_bottom - low) / close
    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std()
    f["volume_ratio_20"] = volume / volume_mean.replace(0, np.nan)
    f["volume_zscore_20"] = (volume - volume_mean) / volume_std.replace(0, np.nan)
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


def _candidates(market_profile: str) -> list[tuple[str, str, dict[str, Any]]]:
    """Small search spaces chosen for stable CPU training, not model-count padding."""
    candidates = [
        ("Ridge (alpha=1)", "Ridge", {"alpha": 1.0}),
        ("Ridge (alpha=10)", "Ridge", {"alpha": 10.0}),
        ("Random Forest", "Random Forest", {"min_samples_leaf": 4}),
        ("Extra Trees", "Extra Trees", {"min_samples_leaf": 4}),
        ("Hist Gradient Boosting (lr=0.05)", "Hist Gradient Boosting", {"learning_rate": 0.05}),
        ("Hist Gradient Boosting (lr=0.10)", "Hist Gradient Boosting", {"learning_rate": 0.10}),
        ("SVR (C=1)", "SVR", {"C": 1.0}),
    ]
    if market_profile == "india":
        candidates.extend([
            ("India Trend Booster (slow)", "Hist Gradient Boosting", {"learning_rate": 0.03, "max_leaf_nodes": 31, "l2_regularization": 5.0}),
        ])
    return candidates


def _model(family: str, parameters: dict[str, Any]):
    if family == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(**parameters))
    if family == "Random Forest":
        return RandomForestRegressor(n_estimators=60, max_features=0.8, random_state=17, n_jobs=1, **parameters)
    if family == "Extra Trees":
        return ExtraTreesRegressor(n_estimators=60, max_features=0.8, random_state=17, n_jobs=1, **parameters)
    if family == "Hist Gradient Boosting":
        defaults = {"max_iter": 120, "max_leaf_nodes": 15, "l2_regularization": 2.0, "random_state": 17}
        return HistGradientBoostingRegressor(**(defaults | parameters))
    if family == "SVR":
        # Approximate the RBF feature map to keep intraday histories tractable.
        return make_pipeline(
            StandardScaler(),
            Nystroem(kernel="rbf", n_components=100, random_state=17),
            LinearSVR(epsilon=0.1, max_iter=4000, random_state=17, dual="auto", **parameters),
        )
    raise ValueError(f"Unknown model family: {family}")


def _fit_recent(estimator, x: pd.DataFrame, y: np.ndarray):
    """Train on a rolling window to reduce cost and emphasize recent regimes."""
    if len(x) > MAX_TRAINING_ROWS:
        x, y = x.iloc[-MAX_TRAINING_ROWS:], y[-MAX_TRAINING_ROWS:]
    estimator.fit(x, y)
    return estimator


def forecast(data: pd.DataFrame, horizon: int = 1, model_choice: str = "auto", market_profile: str = "global_equity") -> ForecastResult:
    if horizon not in (1, 3, 5, 10, 20, 30):
        raise ValueError("Horizon must be 1, 3, 5, 10, 20, or 30 trading bars.")
    choices = ("auto", "ridge", "random forest", "extra trees", "hist gradient boosting", "svr", "weighted ensemble")
    if model_choice.lower() not in choices:
        raise ValueError(f"Model must be one of: {', '.join(choices)}.")
    if len(data) < 250:
        raise ValueError("At least 250 valid price rows are required to calculate the 200-bar trend features and evaluate a forecast.")

    diagnostics, risk = analyze_market(data, horizon)
    features = build_features(data)
    future_price = data["close"].shift(-horizon).rename("future_price")
    target_return = (future_price / data["close"] - 1).rename("target_return")
    dataset = features.join([target_return, future_price]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(dataset) < 80:
        raise ValueError("Insufficient complete feature rows for nested chronological evaluation.")
    x = dataset.drop(columns=["target_return", "future_price"])
    y = dataset["target_return"].to_numpy(dtype=float)
    actual_price = dataset["future_price"].to_numpy(dtype=float)
    prior_close = data["close"].reindex(dataset.index).to_numpy(dtype=float)

    # Reserve the newest contiguous window as a final holdout. Model/hyperparameter
    # selection uses only earlier expanding-window folds; the holdout stays unseen.
    holdout_size = min(40, max(20, len(dataset) // 5))
    holdout_start = len(dataset) - holdout_size
    development_end = holdout_start - horizon
    development_x, development_y = x.iloc[:development_end], y[:development_end]
    fold_size = min(20, max(12, len(development_x) // 5))
    if len(development_x) < 40 + 2 * fold_size + 2 * horizon:
        raise ValueError("Not enough history for purged model-selection folds and a separate holdout at this horizon.")
    splitter = TimeSeriesSplit(n_splits=2, test_size=fold_size, gap=horizon)

    candidates = _candidates(market_profile)
    selection_mae: dict[str, float] = {}
    selection_seconds: dict[str, float] = {}
    candidate_oof: dict[str, list[float]] = {}
    selection_actual: list[float] = []
    for label, family, params in candidates:
        errors = []
        oof_prediction: list[float] = []
        elapsed = 0.0
        for train_idx, valid_idx in splitter.split(development_x):
            estimator = _model(family, params)
            fit_started = perf_counter()
            _fit_recent(estimator, development_x.iloc[train_idx], development_y[train_idx])
            estimate = estimator.predict(development_x.iloc[valid_idx])
            elapsed += perf_counter() - fit_started
            errors.append(mean_absolute_error(development_y[valid_idx], estimate))
            oof_prediction.extend(estimate.tolist())
        candidate_oof[label] = oof_prediction
        selection_mae[label] = float(np.mean(errors))
        selection_seconds[label] = elapsed
    for _, valid_idx in splitter.split(development_x):
        selection_actual.extend(development_y[valid_idx].tolist())

    best_by_family: dict[str, tuple[str, dict[str, Any]]] = {}
    for label, family, params in candidates:
        if family not in best_by_family or selection_mae[label] < selection_mae[best_by_family[family][0]]:
            best_by_family[family] = (label, params)
    family_choice = model_choice.lower()
    if family_choice == "auto":
        selected_family = min(best_by_family, key=lambda name: selection_mae[best_by_family[name][0]])
    elif family_choice == "weighted ensemble":
        selected_family = "Weighted Ensemble"
    else:
        selected_family = next(family for family in best_by_family if family.lower() == family_choice)

    # Keep the final holdout beyond a horizon-sized purge from all training labels.
    holdout_train_end = holdout_start - horizon
    holdout_x, holdout_y = x.iloc[holdout_start:], actual_price[holdout_start:]
    holdout_previous = prior_close[holdout_start:]
    holdout_predictions: dict[str, np.ndarray] = {}
    holdout_seconds: dict[str, float] = {}
    for family, (label, params) in best_by_family.items():
        estimator = _model(family, params)
        fit_started = perf_counter()
        _fit_recent(estimator, x.iloc[:holdout_train_end], y[:holdout_train_end])
        predicted_return = estimator.predict(holdout_x)
        holdout_seconds[family] = perf_counter() - fit_started
        holdout_predictions[family] = holdout_previous * (1 + predicted_return)

    # Blend weights use only the earlier model-selection folds. The final holdout
    # measures the blend after those weights have been fixed.
    raw_weights = {family: 1 / max(selection_mae[label], 1e-9) for family, (label, _) in best_by_family.items()}
    weight_total = sum(raw_weights.values())
    weights = {family: weight / weight_total for family, weight in raw_weights.items()}
    ensemble_oof = sum(np.asarray(candidate_oof[label]) * weights[family] for family, (label, _) in best_by_family.items())
    ensemble_selection_mae = float(mean_absolute_error(selection_actual, ensemble_oof))
    naive_selection_mae = float(mean_absolute_error(selection_actual, np.zeros(len(selection_actual))))
    ensemble_return = sum(
        # Model outputs in this section are already converted to prices. The
        # previous close is common across models, so weighted price blending is
        # equivalent to blending their horizon returns.
        holdout_predictions[family] * weight for family, weight in weights.items()
    )
    holdout_predictions["Weighted Ensemble"] = ensemble_return
    holdout_predictions["Naïve: unchanged close"] = holdout_previous.copy()

    comparison: dict[str, dict[str, float]] = {}
    for family, prediction in holdout_predictions.items():
        row = _metrics(holdout_y, prediction, holdout_previous)
        if family == "Weighted Ensemble":
            row["training_seconds"] = sum(holdout_seconds.values()) + sum(selection_seconds[label] for label, _ in best_by_family.values())
        elif family == "Naïve: unchanged close":
            row["training_seconds"] = 0.0
        else:
            row["training_seconds"] = holdout_seconds[family] + selection_seconds[best_by_family[family][0]]
        if family == "Weighted Ensemble":
            row["selection_mae"] = 100 * ensemble_selection_mae
        elif family == "Naïve: unchanged close":
            row["selection_mae"] = 100 * naive_selection_mae
        else:
            row["selection_mae"] = 100 * selection_mae[best_by_family[family][0]]
        comparison[family] = row
    naive = _metrics(holdout_y, holdout_previous, holdout_previous)
    if market_profile == "forex" and model_choice.lower() == "auto":
        selected_family = "Weighted Ensemble" if ensemble_selection_mae < naive_selection_mae * (1 - MIN_SELECTION_EDGE) else "Naïve: unchanged close"
    elif model_choice.lower() == "auto":
        selected_family = min(best_by_family, key=lambda name: selection_mae[best_by_family[name][0]])
        if selection_mae[best_by_family[selected_family][0]] >= naive_selection_mae * (1 - MIN_SELECTION_EDGE):
            selected_family = "Naïve: unchanged close"
    selected_holdout = holdout_predictions[selected_family]
    metrics = comparison[selected_family]
    if selected_family == "Weighted Ensemble":
        selection_prediction = ensemble_oof
    elif selected_family == "Naïve: unchanged close":
        selection_prediction = np.zeros(len(selection_actual))
    else:
        selection_prediction = np.asarray(candidate_oof[best_by_family[selected_family][0]])
    calibration_errors = np.asarray(selection_actual) - np.asarray(selection_prediction)
    holdout_actual_returns = holdout_y / holdout_previous - 1
    holdout_prediction_returns = selected_holdout / holdout_previous - 1
    holdout_errors = holdout_actual_returns - holdout_prediction_returns

    # Refit chosen model(s) on every label now available, then forecast from the
    # latest feature row. The holdout is used only to report historical errors.
    latest_features = features.iloc[[-1]].replace([np.inf, -np.inf], np.nan)
    if latest_features.isna().any(axis=None):
        raise ValueError("Latest bar does not have enough history for all features.")
    if selected_family == "Weighted Ensemble":
        predicted_return = 0.0
        for family, weight in weights.items():
            _, params = best_by_family[family]
            estimator = _model(family, params)
            fit_started = perf_counter()
            _fit_recent(estimator, x, y)
            predicted_return += float(estimator.predict(latest_features)[0]) * weight
            metrics["training_seconds"] += perf_counter() - fit_started
    elif selected_family == "Naïve: unchanged close":
        predicted_return = 0.0
    else:
        _, params = best_by_family[selected_family]
        estimator = _model(selected_family, params)
        fit_started = perf_counter()
        _fit_recent(estimator, x, y)
        predicted_return = float(estimator.predict(latest_features)[0])
        metrics["training_seconds"] += perf_counter() - fit_started
    selected_model = selected_family if selected_family in ("Weighted Ensemble", "Naïve: unchanged close") else best_by_family[selected_family][0]
    reference_price = float(data["close"].iloc[-1])
    value = max(0.0, reference_price * (1 + predicted_return))

    intervals: dict[str, dict[str, float]] = {}
    for label, tail_probability in (("80", 0.10), ("90", 0.05), ("95", 0.025)):
        lower_error, upper_error = np.quantile(calibration_errors, [tail_probability, 1 - tail_probability])
        lower = max(0.0, reference_price * (1 + predicted_return + lower_error))
        upper = max(lower, reference_price * (1 + predicted_return + upper_error))
        coverage = float(np.mean((holdout_errors >= lower_error) & (holdout_errors <= upper_error)) * 100)
        intervals[label] = {"lower": float(lower), "upper": float(upper), "holdout_coverage_pct": coverage}
    probability_positive = float(np.mean(predicted_return + calibration_errors > 0) * 100)
    target_positions = data.index.get_indexer(dataset.index[holdout_start:]) + horizon
    validation_predictions = [
        {
            "timestamp": data.index[position].isoformat() if hasattr(data.index[position], "isoformat") else str(data.index[position]),
            "actual": float(holdout_y[offset]),
            "predicted": float(selected_holdout[offset]),
            "unchanged_close": float(holdout_previous[offset]),
        }
        for offset, position in enumerate(target_positions)
    ]
    return ForecastResult(
        forecast=value,
        reference_price=reference_price,
        horizon=horizon,
        market_profile=market_profile,
        model=selected_model,
        metrics=metrics,
        naive_metrics=naive,
        model_comparison=comparison,
        ensemble_weights=weights if selected_family == "Weighted Ensemble" or (market_profile == "forex" and model_choice.lower() == "auto") else {},
        diagnostics=diagnostics,
        risk=risk,
        uncertainty={
            "lower": intervals["90"]["lower"],
            "upper": intervals["90"]["upper"],
            "intervals": intervals,
            "method": "Return-residual quantiles from selection folds; coverage measured on the untouched holdout.",
        },
        training_seconds=float(metrics["training_seconds"]),
        return_forecast_pct=float(predicted_return * 100),
        probability_positive=probability_positive,
        validation_predictions=validation_predictions,
        validation_points=len(holdout_y),
        validation_end=str(data.index[-1].isoformat() if hasattr(data.index[-1], "isoformat") else data.index[-1]),
    )

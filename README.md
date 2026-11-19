# GreenToBuy

GreenToBuy is a market analysis application built with FastAPI and Streamlit. It loads Yahoo Finance OHLCV history, calculates technical features, compares regression and tree models with chronological validation, and displays a direct forecast for a selected horizon.

Forecasts are estimates from historical data. They do not account for future news, corporate actions beyond the source data, or changing market conditions, and are not investment advice.

## Features

- Daily and intraday OHLCV history with a local SQLite cache.
- Data validation for missing prices, duplicate timestamps, and invalid symbols.
- Moving averages, EMA, RSI, MACD, Bollinger bands, ATR, ADX, OBV, returns, volatility, momentum, and lag features.
- Ridge, Random Forest, Extra Trees, histogram gradient boosting, and support vector regression, with small validation-based parameter searches.
- Automatic model selection, plus a weighted ensemble whose weights come from earlier validation folds.
- Purged chronological model-selection folds and a separate final holdout, with error metrics, directional accuracy, and an unchanged-close benchmark.
- A Streamlit dashboard with candlesticks, forecast range, latest indicators, and validation results.
- Curated Indian listings and NIFTY 50, NIFTY Bank, and SENSEX indices through Yahoo Finance symbols.
- India-profile trend and volume feature tuning; forex auto mode evaluates a validation-weighted model blend with a baseline guard.

## Architecture

```text
Streamlit dashboard (app.py)
          │ HTTP
FastAPI service (server.py)
       ┌──┴──────────┐
Market data       Forecasting
(market_data.py)  (forecasting.py)
       │               │
Yahoo Finance    pandas / scikit-learn
SQLite cache
```

`market_data.py` handles ticker validation, Yahoo Finance downloads, OHLCV checks, retries, and SQLite caching. `forecasting.py` builds features using information available at each bar, evaluates models chronologically, and fits the selected model for the requested horizon. The UI calls the API and renders only values returned by the service.

## Requirements and setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Start the API and dashboard in separate terminals:

```bash
uvicorn server:app --reload
streamlit run app.py
```

Open <http://localhost:8501>. The dashboard expects the API at `http://127.0.0.1:8000`. Set `GREENTOBUY_API_URL` to use another API address. The API health check is available at `/health`; interactive API documentation is at `/docs`.

## Forecasting and evaluation

The dashboard ticker picker searches a curated list of US equities, ETFs, Indian equities, Indian indices, and supported forex aliases. Indian listings use Yahoo Finance `.NS`/`.BO` symbols; a separate field accepts other Yahoo Finance symbols. The API accepts `interval` (`daily`, `60min`, `15min`, or `5min`), `horizon` (1, 5, 10, or 30 trading bars), and `model` (`auto`, `weighted ensemble`, `ridge`, `random forest`, `extra trees`, `hist gradient boosting`, or `svr`). For example:

```text
/api/stock/AAPL?interval=daily&horizon=5&model=auto
```

Models predict horizon returns from current and lagged price, volume, volatility, candle, and trend features, then convert the return estimate back to a price. India symbols also evaluate regularized trend-boosting variants. Forex symbols in auto mode evaluate a weighted blend of Ridge, Random Forest, Extra Trees, histogram gradient boosting, and approximate RBF SVR; blend weights are fixed from earlier folds. Auto mode uses a model only when selection folds show at least a 5% MAE improvement over the unchanged-close baseline; otherwise it returns that baseline. The kernel approximation and a 1,500-row rolling fit window keep computation bounded for longer intraday histories. Expanding-window selection folds are chronological and include a gap equal to the forecast horizon, so training labels do not overlap validation periods. Model family and parameter selection use only earlier folds; displayed performance uses a later, separate holdout. Results include MAE, RMSE, MAPE, R², directional accuracy, and an unchanged-close benchmark. A 90th percentile absolute holdout error range is shown; it is not a calibrated probability interval.

No model is assumed to be best, and there is no claim that these forecasts beat the benchmark or will be profitable. The forecasting code requires enough history for the 200-bar feature, selection folds, and separate holdout; short history returns a clear error.

## Tests

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

The tests use deterministic synthetic prices and do not call external data services. Live Yahoo Finance availability and rate limits depend on the upstream service.

## Limitations and next steps

- Yahoo Finance is an unofficial data source and may throttle or change its response.
- Indian exchange symbols and history are sourced from Yahoo Finance; coverage and intraday availability can vary by symbol.
- Cached history may be returned if a refresh fails; the API timestamp identifies the last market bar.
- Intraday horizons are expressed in bars and do not account for exchange holidays or market session boundaries.
- The validation error range is descriptive and should not be interpreted as a confidence interval.
- Additional model families should be added only when walk-forward results show measurable value.

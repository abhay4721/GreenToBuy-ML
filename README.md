# GreenToBuy

GreenToBuy is a Streamlit market analysis app. It downloads Yahoo Finance OHLCV data directly through `yfinance`, calculates technical features, compares regression and tree models with chronological validation, and displays a forecast for a selected horizon. The dashboard does not need a local API server or a market-data API key.

Forecasts are estimates from historical data. They do not account for future news, corporate actions beyond the source data, or changing market conditions, and are not investment advice.

## Features

- Daily and intraday OHLCV history with a local SQLite cache.
- Data validation for missing prices, duplicate timestamps, and invalid symbols.
- Moving averages, EMA, RSI, MACD, Bollinger bands, ATR, ADX, OBV, returns, volatility, momentum, and lag features.
- Ridge, Random Forest, Extra Trees, histogram gradient boosting, and support vector regression, with small validation-based parameter searches.
- A regularized stacked ensemble across the supported model families for every market profile; Auto uses it only if earlier out-of-fold predictions clear the unchanged-close baseline hurdle.
- Purged chronological model-selection folds and a separate final holdout, with error metrics, directional accuracy, and an unchanged-close benchmark.
- Six forecast horizons (1, 3, 5, 10, 20, and 30 bars), return-residual probability summaries, and 80/90/95% empirical intervals with separate holdout coverage reporting.
- Causal return, drawdown, Hurst, trend-regime, EWMA, Parkinson, and Garman-Klass diagnostics with explicit assumptions.
- A local reproducibility ledger recording forecast settings, feature set, software versions, random seed, metrics, and recent run history.
- Pointwise predictions from the final chronological holdout, plotted with actual prices and the unchanged-close baseline.
- A Streamlit dashboard with candlesticks, forecast range, latest indicators, and validation results.
- Curated Indian listings and NIFTY 50, NIFTY Bank, and SENSEX indices through Yahoo Finance symbols.
- India-profile trend and volume feature tuning; its additional trend-boosting candidate participates in the same stack. Forex and other markets use the same family stack with weights fitted separately per series.
- Latest Yahoo Finance five-minute bar with local paper orders for supported instruments.

## Architecture

```text
Streamlit dashboard (app.py)
       │ direct Python calls
       ├── Yahoo Finance data (market_data.py / yfinance)
       ├── Forecasting (forecasting.py / scikit-learn)
       └── Local SQLite stores
```

`market_data.py` handles ticker validation, direct Yahoo Finance downloads, OHLCV checks, retries, and SQLite caching. `local_analysis.py` calls data acquisition, forecasting, diagnostics, and research logging in-process. `forecasting.py` builds features using information available at each bar, evaluates models chronologically, and fits the selected model for the requested horizon. `quant_analysis.py` computes descriptive diagnostics from historical bars. `research_store.py` stores completed run metadata and scores in a separate, ignored `research_history.db`.

`paper_trading.py` records simulated fills and positions in a separate, ignored `paper_trading.db`; it never submits orders to a broker. The quote panel refreshes the latest Yahoo Finance five-minute bar once a minute. Yahoo Finance may delay intraday data and does not provide an executable bid/ask.

## Requirements and setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Start the backend and frontend together on Linux:

```bash
chmod +x run_linux.sh   # only needed once
./run_linux.sh
```

Open <http://localhost:8501>. The launcher starts Uvicorn at `http://127.0.0.1:8000` and Streamlit at `http://127.0.0.1:8501`; pressing Ctrl+C stops both. Set `BACKEND_PORT` or `FRONTEND_PORT` to change ports, and set `BACKEND_HOST` or `FRONTEND_HOST` to change bind addresses. The Streamlit dashboard retrieves market history from Yahoo Finance directly; its analysis does not call the local backend.

To run only the dashboard, use `streamlit run app.py`. The local API server remains useful for other clients and is started by `run_linux.sh` as requested.

The quote panel supports **paper orders only**. It uses the latest Yahoo Finance five-minute close as a simulated fill and rejects bars older than 60 seconds; Yahoo may delay or omit intraday data, so paper orders may be unavailable. Positions and orders remain local in `paper_trading.db`; no broker order is sent.

## Forecasting and evaluation

The dashboard ticker picker searches a curated list of US equities, ETFs, Indian equities, Indian indices, and supported forex aliases. Indian listings use Yahoo Finance `.NS`/`.BO` symbols; a separate field accepts other Yahoo Finance symbols. Choose an interval (`daily`, `60min`, `15min`, or `5min`), horizon (1, 3, 5, 10, 20, or 30 bars), and model in the sidebar. Each analysis requests fresh history from Yahoo Finance, falling back to the local cache if the source is unavailable. `server.py` provides an optional API for integrations, but is not part of the dashboard's data path.

For integrations that choose to run the optional HTTP service:

```text
/api/stock/AAPL?interval=daily&horizon=5&model=auto
```

Models predict horizon returns from current and lagged price, volume, volatility, candle, and trend features, then convert the return estimate back to a price. The stacked model combines the best validated Ridge, Random Forest, Extra Trees, histogram gradient boosting, and approximate RBF SVR variants. Its nonnegative weights are fit with regularization toward equal weights using only chronological out-of-fold predictions; the newest holdout remains separate. Indian symbols add a regularized trend-boosting candidate to the histogram gradient boosting family. Auto mode uses the stack only when its selection-fold MAE clears the unchanged-close baseline by at least 5%; otherwise it returns that baseline. This validation guard can help avoid choosing a model that has not shown an edge, but cannot make future predictions reliably accurate in every market. The kernel approximation and a 1,500-row rolling fit window keep computation bounded for longer intraday histories. Expanding-window selection folds are chronological and include a gap equal to the forecast horizon, so training labels do not overlap validation periods. Model family and parameter selection use only earlier folds; displayed performance uses a later, separate holdout. Results include MAE, RMSE, MAPE, R², directional accuracy, training duration, and an unchanged-close benchmark. Return-residual quantiles from the selection folds produce empirical 80/90/95% intervals; their measured coverage is reported on the separate holdout. The probability-of-positive figure is an empirical residual frequency, not a calibrated probability guarantee. Every successful analysis is recorded in the local run ledger for later review. This run ledger does not yet match forecasts to matured outcomes automatically.

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
- The regime label is a transparent trailing-window heuristic, not a trained regime classifier. Hurst and volatility estimators are descriptive and have known sampling assumptions.
- Empirical residual interval coverage is measured on a finite holdout; it can differ from the nominal interval levels and is not a guarantee.
- Additional model families should be added only when walk-forward results show measurable value.

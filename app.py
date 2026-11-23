"""Streamlit dashboard for historical market data and model evaluation."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from local_analysis import analyze_symbol
from market_data import get_stock_data, market_profile
from paper_trading import order_history, place_paper_order, positions
from research_store import recent_experiments

st.set_page_config(page_title="GreenToBuy | Market Analytics", page_icon="📈", layout="wide")
TICKERS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon", "GOOGL": "Alphabet",
    "META": "Meta Platforms", "TSLA": "Tesla", "AVGO": "Broadcom", "BRK-B": "Berkshire Hathaway",
    "JPM": "JPMorgan Chase", "LLY": "Eli Lilly", "V": "Visa", "XOM": "Exxon Mobil", "MA": "Mastercard",
    "COST": "Costco", "WMT": "Walmart", "ORCL": "Oracle", "NFLX": "Netflix", "AMD": "AMD",
    "CRM": "Salesforce", "QCOM": "Qualcomm", "ADBE": "Adobe", "INTC": "Intel", "CSCO": "Cisco",
    "IBM": "IBM", "GE": "GE Aerospace", "CAT": "Caterpillar", "BA": "Boeing", "DIS": "Walt Disney",
    "KO": "Coca-Cola", "PEP": "PepsiCo", "NKE": "Nike", "MCD": "McDonald's", "UBER": "Uber",
    "PLTR": "Palantir", "COIN": "Coinbase", "SHOP": "Shopify", "PYPL": "PayPal", "BABA": "Alibaba",
    "TSM": "Taiwan Semiconductor", "SONY": "Sony", "SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco QQQ ETF",
    "DIA": "SPDR Dow Jones ETF", "IWM": "iShares Russell 2000 ETF", "VTI": "Vanguard Total Market ETF",
    "VOO": "Vanguard S&P 500 ETF", "GLD": "SPDR Gold Shares", "SLV": "iShares Silver Trust",
    "TLT": "iShares 20+ Year Treasury ETF", "XLF": "Financial Select Sector ETF",
    "XLE": "Energy Select Sector ETF", "XAUUSD": "Gold / US Dollar", "EURUSD": "Euro / US Dollar",
    "GBPUSD": "British Pound / US Dollar",
    "RELIANCE.NS": "Reliance Industries · NSE", "TCS.NS": "Tata Consultancy Services · NSE",
    "HDFCBANK.NS": "HDFC Bank · NSE", "ICICIBANK.NS": "ICICI Bank · NSE", "INFY.NS": "Infosys · NSE",
    "SBIN.NS": "State Bank of India · NSE", "BHARTIARTL.NS": "Bharti Airtel · NSE",
    "ITC.NS": "ITC Limited · NSE", "LT.NS": "Larsen & Toubro · NSE", "HINDUNILVR.NS": "Hindustan Unilever · NSE",
    "AXISBANK.NS": "Axis Bank · NSE", "KOTAKBANK.NS": "Kotak Mahindra Bank · NSE",
    "MARUTI.NS": "Maruti Suzuki · NSE", "TATAMOTORS.NS": "Tata Motors · NSE",
    "TATASTEEL.NS": "Tata Steel · NSE", "SUNPHARMA.NS": "Sun Pharmaceutical · NSE",
    "BAJFINANCE.NS": "Bajaj Finance · NSE", "ASIANPAINT.NS": "Asian Paints · NSE",
    "TITAN.NS": "Titan Company · NSE", "ADANIENT.NS": "Adani Enterprises · NSE",
    "NTPC.NS": "NTPC · NSE", "POWERGRID.NS": "Power Grid · NSE", "ONGC.NS": "ONGC · NSE",
    "WIPRO.NS": "Wipro · NSE", "HCLTECH.NS": "HCL Technologies · NSE",
    "ULTRACEMCO.NS": "UltraTech Cement · NSE", "^NSEI": "NIFTY 50 Index",
    "^NSEBANK": "NIFTY Bank Index", "^BSESN": "BSE SENSEX Index",
}

st.markdown("""
<style>
  .stApp { background: #0b1118; color: #e5edf5; }
  [data-testid="stSidebar"] { background: #101923; }
  [data-testid="stMetric"] { background: #111d28; border: 1px solid #243341; padding: 14px; border-radius: 10px; }
  h1, h2, h3 { letter-spacing: -0.025em; }
</style>
""", unsafe_allow_html=True)

st.title("GreenToBuy")
st.caption("Historical market analysis and out-of-sample evaluated forecasts")

with st.sidebar:
    st.subheader("Analysis")
    ticker_options = list(TICKERS)
    selected_ticker = st.selectbox(
        "Search ticker",
        ticker_options,
        index=ticker_options.index("AAPL"),
        format_func=lambda ticker: f"{ticker}  ·  {TICKERS[ticker]}",
        help="Type a symbol or company name to filter the list.",
    )
    custom_ticker = st.text_input("Or enter another ticker", placeholder="e.g. MSFT, BRK-B, XAUUSD", max_chars=20).strip().upper()
    symbol = custom_ticker or selected_ticker
    interval = st.selectbox("Market interval", ["daily", "60min", "15min", "5min"])
    horizon_unit = "trading day" if interval == "daily" else "bar"
    horizon = st.selectbox("Forecast horizon", [1, 3, 5, 10, 20, 30], format_func=lambda x: f"{x} {horizon_unit if x == 1 else horizon_unit + 's'}")
    model_labels = {
        "auto": "All-family stacked model (validated)",
        "weighted ensemble": "Stacked ensemble",
        "ridge": "Ridge regression",
        "random forest": "Random forest",
        "extra trees": "Extra Trees",
        "hist gradient boosting": "Histogram gradient boosting",
        "svr": "RBF support vector regression (approx.)",
    }
    model = st.selectbox("Model selection", list(model_labels), format_func=lambda value: model_labels[value])
    analyze = st.button("Run analysis", type="primary", use_container_width=True)
    st.caption("Forecasts are statistical estimates, not guaranteed future prices.")


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_latest_quote(ticker: str) -> dict:
    data = get_stock_data(ticker, "5min", force_refresh=True)
    timestamp = data.index[-1]
    age_seconds = max(0.0, (datetime.now(timezone.utc) - timestamp.to_pydatetime()).total_seconds())
    return {
        "symbol": ticker,
        "status": "live" if age_seconds <= 60 else "stale",
        "price": float(data["close"].iloc[-1]),
        "timestamp": timestamp.isoformat(),
        "age_seconds": age_seconds,
        "currency": "INR" if market_profile(ticker) == "india" else None,
    }


@st.fragment(run_every=60)
def realtime_paper_panel(ticker: str) -> None:
    st.subheader("Latest Yahoo Finance bar · paper trading")
    try:
        quote = _fetch_latest_quote(ticker)
    except Exception as exc:
        st.info(f"Yahoo Finance quote unavailable: {exc}")
        return

    if quote["status"] != "live":
        st.warning("The latest Yahoo Finance 5-minute bar is older than 60 seconds. Paper orders require a fresh bar.")
    quote_columns = st.columns(2)
    quote_columns[0].metric("Latest price", f"{quote['price']:,.5f}".rstrip("0").rstrip("."))
    quote_columns[1].metric("Bar time (UTC)", quote["timestamp"])
    st.caption("Yahoo Finance intraday data may be delayed and does not provide a guaranteed executable bid/ask. Orders here are simulated locally.")

    if ticker.startswith("^"):
        st.caption("Index quotes are view-only in the paper order panel.")
    else:
        with st.form(f"paper_order_{ticker}"):
            side_col, quantity_col = st.columns(2)
            side = side_col.selectbox("Paper side", ["BUY", "SELL"], key=f"paper_side_{ticker}")
            quantity = quantity_col.number_input("Quantity", min_value=0.000001, value=1.0, key=f"paper_quantity_{ticker}")
            submitted = st.form_submit_button("Place paper order", type="primary")
        if submitted:
            try:
                order_response = place_paper_order(ticker, side, quantity, quote)
                st.success(f"Paper {side.lower()} {quantity:g} {ticker} at {order_response['fill_price']:,.5f}. No broker order was sent.")
            except ValueError as exc:
                st.error(f"Paper order rejected: {exc}")

    paper_positions = positions()
    if paper_positions:
        st.caption("Local simulated positions")
        st.dataframe(pd.DataFrame(paper_positions), hide_index=True, use_container_width=True)
    recent_orders = order_history(10)
    if recent_orders:
        st.caption("Recent local paper orders")
        st.dataframe(pd.DataFrame(recent_orders), hide_index=True, use_container_width=True)


realtime_paper_panel(symbol)

if analyze:
    if not symbol:
        st.error("Enter a ticker symbol.")
        st.stop()
    with st.spinner(f"Loading {symbol} market data and evaluating the model…"):
        try:
            result = analyze_symbol(symbol, interval, horizon, model)
        except Exception as exc:
            st.error(f"Analysis could not be completed: {exc}")
            st.stop()

    forecast = result["forecast"]
    st.caption(f"Research run: {result.get('experiment_id') or 'not saved'}")
    current = result["latest_price"]["price"]
    expected = (forecast["forecast"] / current - 1) * 100 if current else 0
    st.subheader(f"{result['symbol']} · {result['interval']}")
    metrics = st.columns(4)
    metrics[0].metric("Reference close", f"{current:,.2f}")
    metrics[1].metric(f"{horizon}-bar forecast", f"{forecast['forecast']:,.2f}", f"{expected:+.2f}%")
    metrics[2].metric("Model", forecast["model"])
    metrics[3].metric("Holdout MAE", f"{forecast['metrics']['mae']:,.2f}")
    st.caption(f"Source: {result['data_source']} · Market profile: {result['market_profile']} · Latest bar: {result['latest_price']['timestamp']} · Analysis updated: {result['updated_at']}")
    if result["market_profile"] == "forex" and model == "auto":
        st.caption("Forex auto mode stacks the supported model families and uses the blend only when it clears the baseline hurdle.")
    elif result["market_profile"] == "india":
        st.caption("India auto mode includes its regularized trend-boosting candidate in the validation-fitted stack.")
        external_data = result["external_data"]
        if external_data["available"]:
            change = external_data["usd_inr_return_5obs_pct"]
            change_text = f" · 5-observation move: {change:+.2f}%" if change is not None else ""
            st.caption(f"Point-in-time macro feature: USD/INR {external_data['usd_inr_rate']:.4f} (reference {external_data['reference_date']}, available from {external_data['available_from']}; lagged one day){change_text} · Frankfurter")
        else:
            st.caption(f"USD/INR macro feature unavailable; this run used market-price features only. {external_data['reason']}")

    chart = pd.DataFrame(result["chart_data"])
    chart["x"] = pd.to_datetime(chart["x"], utc=True)
    fig = go.Figure(go.Candlestick(x=chart.x, open=chart.o, high=chart.h, low=chart.l, close=chart.c, name=result["symbol"]))
    last_time = chart.x.iloc[-1]
    if interval == "daily":
        future_time = pd.bdate_range(start=last_time.tz_localize(None), periods=horizon + 1)[-1].tz_localize("UTC")
    else:
        future_time = last_time + pd.Timedelta(minutes={"5min": 5, "15min": 15, "60min": 60}[interval] * horizon)
    band = forecast["uncertainty"]
    fig.add_trace(go.Scatter(x=[last_time, future_time], y=[current, forecast["forecast"]], mode="lines+markers", name="Forecast", line={"color": "#42d392", "dash": "dash", "width": 2}))
    fig.add_trace(go.Scatter(x=[future_time, future_time], y=[band["lower"], band["upper"]], mode="lines", name="90% selection-residual interval", line={"color": "#f2b84b", "width": 7}, opacity=0.55))
    fig.update_layout(template="plotly_dark", height=510, xaxis_rangeslider_visible=False, margin={"l": 10, "r": 10, "t": 20, "b": 10}, legend={"orientation": "h", "y": 1.02})
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Model comparison")
        st.caption(f"{forecast['validation_points']} untouched chronological holdout predictions · evaluated through {forecast['validation_end']}")
        score_rows = []
        for name, values in forecast["model_comparison"].items():
            score_rows.append({"Method": name, "Selection MAE (% return)": f"{values['selection_mae']:.3f}", "Holdout MAE": f"{values['mae']:.3f}", "RMSE": f"{values['rmse']:.3f}", "MAPE (%)": f"{values['mape']:.3f}", "R²": f"{values['r2']:.3f}", "Direction (%)": f"{values['directional_accuracy']:.1f}", "Training (s)": f"{values['training_seconds']:.2f}"})
        st.dataframe(pd.DataFrame(score_rows).set_index("Method"), use_container_width=True)
        if forecast["model"] == "Naïve: unchanged close":
            st.info("No candidate cleared the 5% selection-fold improvement hurdle over the unchanged-close benchmark, so the forecast uses that baseline.")
        elif forecast["metrics"]["mae"] >= forecast["naive_metrics"]["mae"]:
            st.info("The selected model did not beat the unchanged-close benchmark on the chronological holdout MAE.")
        if forecast["ensemble_weights"]:
            st.caption("Out-of-fold ensemble weights: " + ", ".join(f"{name} {weight:.0%}" for name, weight in forecast["ensemble_weights"].items()))
        st.caption(f"Selection-residual probability of positive return: {forecast['probability_positive']:.1f}% · Selection interval coverage on holdout: {forecast['uncertainty']['intervals']['90']['holdout_coverage_pct']:.1f}%")
    with right:
        st.subheader("Latest indicators")
        indicators = result["indicators"]
        show = {"SMA 20": "sma_20", "SMA 50": "sma_50", "SMA 200": "sma_200", "RSI 14": "rsi", "MACD": "macd", "ATR 14": "atr", "ADX 14": "adx", "Bollinger upper": "bb_upper", "Bollinger lower": "bb_lower"}
        st.dataframe(pd.DataFrame([{"Indicator": label, "Value": f"{indicators[key]:,.3f}" if indicators[key] is not None else "Not enough history"} for label, key in show.items()]).set_index("Indicator"), use_container_width=True)
        st.caption(f"Return-residual interval method: {band['method']}. Holdout coverage is shown above; it is not a guarantee.")
        st.subheader("Quantitative state and risk")
        st.caption(result["diagnostics"]["regime"])
        st.write(f"Hurst estimate: {result['diagnostics']['hurst_estimate']:.3f}" if result["diagnostics"]["hurst_estimate"] is not None else "Hurst estimate: insufficient history")
        st.write(f"EWMA volatility per bar: {result['risk']['ewma_volatility_pct_per_bar']:.3f}%")
        st.write(f"{horizon}-bar square-root-time volatility estimate: {result['risk']['sqrt_time_horizon_volatility_pct']:.3f}%")
        st.caption(result["risk"]["range_assumption"])

    st.subheader("Untouched holdout: actual vs predicted")
    holdout = pd.DataFrame(forecast["validation_predictions"])
    holdout["timestamp"] = pd.to_datetime(holdout["timestamp"], utc=True)
    holdout_fig = go.Figure()
    holdout_fig.add_trace(go.Scatter(x=holdout["timestamp"], y=holdout["actual"], mode="lines+markers", name="Actual", line={"color": "#e8edf2"}))
    holdout_fig.add_trace(go.Scatter(x=holdout["timestamp"], y=holdout["predicted"], mode="lines+markers", name="Selected model", line={"color": "#42d392"}))
    holdout_fig.add_trace(go.Scatter(x=holdout["timestamp"], y=holdout["unchanged_close"], mode="lines", name="Unchanged-close baseline", line={"color": "#f2b84b", "dash": "dot"}))
    holdout_fig.update_layout(template="plotly_dark", height=360, margin={"l": 10, "r": 10, "t": 10, "b": 10}, legend={"orientation": "h", "y": 1.02})
    st.plotly_chart(holdout_fig, use_container_width=True)

    with st.expander("Recent saved forecast runs"):
        runs = recent_experiments(result["symbol"], limit=20)
        if runs:
            history_rows = [{
                "Run": run["experiment_id"][:8],
                "Time": run["created_at"],
                "Horizon (bars)": run["horizon"],
                "Model": run["selected_model"],
                "Forecast": run["forecast_price"],
                "Return (%)": run["return_forecast_pct"],
                "Holdout MAE": run["metrics"]["selected"]["mae"],
            } for run in runs]
            st.dataframe(pd.DataFrame(history_rows), hide_index=True, use_container_width=True)
            st.caption("Saved forecasts and their historical holdout scores. Future realized outcomes are not yet joined to these runs.")
        else:
            st.caption("No saved runs for this symbol yet.")
else:
    st.info("Choose a ticker and analysis settings, then run an analysis.")

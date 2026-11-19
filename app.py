"""Streamlit dashboard for historical market data and model evaluation."""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="GreenToBuy | Market Analytics", page_icon="📈", layout="wide")
API_URL = os.getenv("GREENTOBUY_API_URL", "http://127.0.0.1:8000")
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
    horizon = st.selectbox("Forecast horizon", [1, 5, 10, 30], format_func=lambda x: f"{x} {horizon_unit if x == 1 else horizon_unit + 's'}")
    model_labels = {
        "auto": "Best model by time-series CV",
        "weighted ensemble": "Validation-weighted ensemble",
        "ridge": "Ridge regression",
        "random forest": "Random forest",
        "extra trees": "Extra Trees",
        "hist gradient boosting": "Histogram gradient boosting",
        "svr": "RBF support vector regression (approx.)",
    }
    model = st.selectbox("Model selection", list(model_labels), format_func=lambda value: model_labels[value])
    analyze = st.button("Run analysis", type="primary", use_container_width=True)
    st.caption("Forecasts are statistical estimates, not guaranteed future prices.")


@st.fragment(run_every=2)
def realtime_paper_panel(ticker: str) -> None:
    st.subheader("Live quote · paper trading")
    previous_ticker = st.session_state.get("_realtime_ticker")
    if previous_ticker and previous_ticker != ticker:
        try:
            requests.delete(f"{API_URL}/api/realtime/quote/{previous_ticker}", timeout=3)
        except requests.RequestException:
            pass
    st.session_state["_realtime_ticker"] = ticker
    try:
        response = requests.get(f"{API_URL}/api/realtime/quote/{ticker}", timeout=4)
        if response.status_code >= 400:
            detail = response.json().get("detail", response.text)
            st.info(f"Live quote unavailable: {detail}")
            return
        quote = response.json()
    except requests.RequestException as exc:
        st.info(f"Live quote API unavailable: {exc}")
        return

    if quote.get("status") == "connecting":
        st.caption("Connecting to the market-data stream…")
        return
    if quote.get("status") != "live":
        st.warning(quote.get("detail", "No live quote received from the provider."))
        return

    quote_columns = st.columns(4)
    quote_columns[0].metric("Live price", f"{quote['price']:,.5f}".rstrip("0").rstrip("."))
    quote_columns[1].metric("Bid", f"{quote['bid']:,.5f}".rstrip("0").rstrip(".") if quote.get("bid") is not None else "—")
    quote_columns[2].metric("Ask", f"{quote['ask']:,.5f}".rstrip("0").rstrip(".") if quote.get("ask") is not None else "—")
    quote_columns[3].metric("Quote time (UTC)", quote["timestamp"])
    st.caption(f"Twelve Data instrument: {quote.get('provider_symbol')} · Exchange: {quote.get('exchange') or 'Forex'}")

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
                order_response = requests.post(
                    f"{API_URL}/api/paper/orders",
                    json={"symbol": ticker, "side": side, "quantity": quantity},
                    timeout=5,
                )
                order_response.raise_for_status()
                st.success(f"Paper {side.lower()} {quantity:g} {ticker} at {order_response.json()['fill_price']:,.5f}. No broker order was sent.")
            except requests.RequestException as exc:
                detail = exc.response.json().get("detail", str(exc)) if exc.response is not None else str(exc)
                st.error(f"Paper order rejected: {detail}")

    try:
        positions_response = requests.get(f"{API_URL}/api/paper/positions", timeout=3)
        positions_response.raise_for_status()
        paper_positions = positions_response.json()
        if paper_positions:
            st.caption("Local simulated positions")
            st.dataframe(pd.DataFrame(paper_positions), hide_index=True, use_container_width=True)
    except requests.RequestException:
        pass


realtime_paper_panel(symbol)

if analyze:
    if not symbol:
        st.error("Enter a ticker symbol.")
        st.stop()
    with st.spinner(f"Loading {symbol} market data and evaluating the model…"):
        try:
            response = requests.get(
                f"{API_URL}/api/stock/{symbol}",
                params={"interval": interval, "horizon": horizon, "model": model},
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            detail = exc.response.json().get("detail", str(exc)) if exc.response is not None else str(exc)
            st.error(f"Analysis could not be completed: {detail}")
            st.stop()

    forecast = result["forecast"]
    current = result["latest_price"]["price"]
    expected = (forecast["forecast"] / current - 1) * 100 if current else 0
    st.subheader(f"{result['symbol']} · {result['interval']}")
    metrics = st.columns(4)
    metrics[0].metric("Reference close", f"{current:,.2f}")
    metrics[1].metric(f"{horizon}-bar forecast", f"{forecast['forecast']:,.2f}", f"{expected:+.2f}%")
    metrics[2].metric("Model", forecast["model"])
    metrics[3].metric("Holdout MAE", f"{forecast['metrics']['mae']:,.2f}")
    st.caption(f"Market profile: {result['market_profile']} · Reference data: {result['latest_price']['timestamp']} · Analysis updated: {result['updated_at']}")
    if result["market_profile"] == "forex" and model == "auto":
        st.caption("Forex auto mode evaluates a weighted blend and uses it only when it clears the baseline hurdle.")
    elif result["market_profile"] == "india":
        st.caption("India profile compares the general model set with regularized trend-boosting variants using earlier validation folds.")

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
    fig.add_trace(go.Scatter(x=[future_time, future_time], y=[band["lower"], band["upper"]], mode="lines", name="90% validation error range", line={"color": "#f2b84b", "width": 7}, opacity=0.55))
    fig.update_layout(template="plotly_dark", height=510, xaxis_rangeslider_visible=False, margin={"l": 10, "r": 10, "t": 20, "b": 10}, legend={"orientation": "h", "y": 1.02})
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Model comparison")
        st.caption(f"{forecast['validation_points']} untouched chronological holdout predictions · evaluated through {forecast['validation_end']}")
        score_rows = []
        for name, values in forecast["model_comparison"].items():
            score_rows.append({"Method": name, "Selection MAE (% return)": f"{values['selection_mae']:.3f}", "Holdout MAE": f"{values['mae']:.3f}", "RMSE": f"{values['rmse']:.3f}", "MAPE (%)": f"{values['mape']:.3f}", "R²": f"{values['r2']:.3f}", "Direction (%)": f"{values['directional_accuracy']:.1f}"})
        st.dataframe(pd.DataFrame(score_rows).set_index("Method"), use_container_width=True)
        if forecast["model"] == "Naïve: unchanged close":
            st.info("No candidate cleared the 5% selection-fold improvement hurdle over the unchanged-close benchmark, so the forecast uses that baseline.")
        elif forecast["metrics"]["mae"] >= forecast["naive_metrics"]["mae"]:
            st.info("The selected model did not beat the unchanged-close benchmark on the chronological holdout MAE.")
        if forecast["ensemble_weights"]:
            st.caption("Forex validation blend weights: " + ", ".join(f"{name} {weight:.0%}" for name, weight in forecast["ensemble_weights"].items()))
    with right:
        st.subheader("Latest indicators")
        indicators = result["indicators"]
        show = {"SMA 20": "sma_20", "SMA 50": "sma_50", "SMA 200": "sma_200", "RSI 14": "rsi", "MACD": "macd", "ATR 14": "atr", "ADX 14": "adx", "Bollinger upper": "bb_upper", "Bollinger lower": "bb_lower"}
        st.dataframe(pd.DataFrame([{"Indicator": label, "Value": f"{indicators[key]:,.3f}" if indicators[key] is not None else "Not enough history"} for label, key in show.items()]).set_index("Indicator"), use_container_width=True)
        st.caption(f"Forecast range uses the {band['method']}; it is a validation error range, not a calibrated probability interval.")
else:
    st.info("Choose a ticker and analysis settings, then run an analysis.")

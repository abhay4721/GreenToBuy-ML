"""Streamlit dashboard for historical market data and model evaluation."""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="GreenToBuy | Market Analytics", page_icon="📈", layout="wide")
API_URL = os.getenv("GREENTOBUY_API_URL", "http://127.0.0.1:8000")

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
    symbol = st.text_input("Ticker", value="AAPL", max_chars=20).strip().upper()
    interval = st.selectbox("Market interval", ["daily", "60min", "15min", "5min"])
    horizon = st.selectbox("Forecast horizon", [1, 5, 10, 30], format_func=lambda x: f"{x} trading {'day' if x == 1 else 'days'}")
    model = st.selectbox("Model selection", ["auto", "ridge", "random forest"], format_func=lambda x: {"auto": "Choose by validation MAE", "ridge": "Ridge regression", "random forest": "Random forest"}[x])
    analyze = st.button("Run analysis", type="primary", use_container_width=True)
    st.caption("Forecasts are statistical estimates, not guaranteed future prices.")

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
    metrics[3].metric("Walk-forward MAE", f"{forecast['metrics']['mae']:,.2f}")
    st.caption(f"Reference data: {result['latest_price']['timestamp']} · Analysis updated: {result['updated_at']}")

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
        st.subheader("Walk-forward performance")
        st.caption(f"{forecast['validation_points']} out-of-sample predictions · evaluated through {forecast['validation_end']}")
        score_rows = []
        for name, values in ((forecast["model"], forecast["metrics"]), ("Naïve: unchanged close", forecast["naive_metrics"])):
            score_rows.append({"Method": name, "MAE": values["mae"], "RMSE": values["rmse"], "MAPE (%)": values["mape"], "R²": values["r2"], "Direction (%)": values["directional_accuracy"]})
        st.dataframe(pd.DataFrame(score_rows).set_index("Method").style.format("{:.3f}"), use_container_width=True)
        if forecast["metrics"]["mae"] >= forecast["naive_metrics"]["mae"]:
            st.info("The selected model did not beat the unchanged-close benchmark on walk-forward MAE.")
    with right:
        st.subheader("Latest indicators")
        indicators = result["indicators"]
        show = {"SMA 20": "sma_20", "SMA 50": "sma_50", "SMA 200": "sma_200", "RSI 14": "rsi", "MACD": "macd", "ATR 14": "atr", "ADX 14": "adx", "Bollinger upper": "bb_upper", "Bollinger lower": "bb_lower"}
        st.dataframe(pd.DataFrame([{"Indicator": label, "Value": f"{indicators[key]:,.3f}" if indicators[key] is not None else "Not enough history"} for label, key in show.items()]).set_index("Indicator"), use_container_width=True)
        st.caption(f"Forecast range uses the {band['method']}; it is a validation error range, not a calibrated probability interval.")
else:
    st.info("Choose a ticker and analysis settings, then run an analysis.")

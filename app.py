# app.py
import talib
import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="GreenToBuy", layout="wide")

st.title("GreenToBuy")
st.markdown("Enter a stock or forex symbol (e.g., AAPL, XAUUSD) to get price predictions and analysis.")

symbol = st.sidebar.text_input("Enter Symbol (e.g., AAPL, XAUUSD)", value="XAUUSD").upper()
timeframe = st.sidebar.selectbox("Select Timeframe", ["5 Minutes", "15 Minutes", "1 Hour", "Daily"], index=0)
model_type = st.sidebar.selectbox("Select Model", ["LSTM", "XGBoost"], index=0)

timeframe_map = {
    "5 Minutes": "5min",
    "15 Minutes": "15min",
    "1 Hour": "60min",
    "Daily": "daily"
}
selected_interval = timeframe_map[timeframe]

if st.sidebar.button("Analyze"):
    try:
        response = requests.get(f"http://localhost:8000/api/stock/{symbol}?model_type={model_type.lower()}")
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"{symbol} Candlestick Chart ({timeframe})")
        chart_data = data.get("chart_data", {}).get(selected_interval, [])
        
        if not chart_data or isinstance(chart_data, dict) and "error" in chart_data:
            st.warning(f"No data available for {symbol} ({timeframe})")
        else:
            df = pd.DataFrame(chart_data)
            df['x'] = pd.to_datetime(df['x'])
            
            fig = go.Figure(data=[
                go.Candlestick(
                    x=df['x'],
                    open=df['o'],
                    high=df['h'],
                    low=df['l'],
                    close=df['c'],
                    name=symbol
                )
            ])
            fig.update_layout(
                title=f"{symbol} ({timeframe})",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Current Price")
        latest_price = data.get("latest_price", {})
        if latest_price and isinstance(latest_price, dict) and "price" in latest_price:
            st.write(f"Price: ${latest_price['price']:.2f}")
            st.write(f"As of: {latest_price['timestamp']}")
        else:
            st.write("Current price not available")
        
        st.subheader("Predictions")
        predictions = data.get("predictions", {})
        
        for interval, prediction in predictions.items():
            timeframe_display = {
                "5min": "5 Minutes",
                "15min": "15 Minutes",
                "60min": "1 Hour",
                "daily": "Next Day"
            }.get(interval, interval)
            
            if isinstance(prediction, dict) and "error" in prediction:
                st.write(f"{timeframe_display}: {prediction['error']}")
            else:
                st.write(f"{timeframe_display}: ${prediction:.2f}")
        
        st.subheader("Sentiment Analysis")
        sentiment = data.get("sentiment", 0.0)
        sentiment_label = "Neutral" if abs(sentiment) < 0.1 else "Bullish" if sentiment > 0 else "Bearish"
        st.write(f"Sentiment: {sentiment_label} ({sentiment:.2f})")
        
        st.subheader("Candlestick Patterns")
        patterns = data.get("patterns", {}).get(selected_interval, [])
        if not patterns:
            st.write("No patterns detected")
        else:
            for pattern in patterns[-5:]:
                st.write(f"{pattern['date']}: {pattern['pattern']} ({pattern['signal']})")
        
        st.write(f"Model Used: {data.get('model_type', 'LSTM').upper()}")

# Disclaimer
st.markdown("---")
st.markdown("**Disclaimer**: This is for informational purposes only. Predictions are not 100% accurate and should not be used as financial advice.")
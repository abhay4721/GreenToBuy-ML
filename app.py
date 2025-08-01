import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("AI-Powered Stock Price Predictor")
st.markdown("Analyze stock prices with advanced machine learning, candlestick patterns, and news sentiment.")

# Sidebar for input
st.sidebar.header("Stock Analysis")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL").upper()
time_period = st.sidebar.selectbox("Select Time Period", ["30 Days"])
analyze_button = st.sidebar.button("Analyze Stock")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.patterns = []
    st.session_state.prediction = None
    st.session_state.sentiment = None
    st.session_state.error = None

if analyze_button:
    with st.spinner("Fetching and analyzing data..."):
        try:
            response = requests.get(f"http://localhost:8000/api/stock/{symbol}")
            response.raise_for_status()
            result = response.json()
            
            st.session_state.data = result['data']
            st.session_state.patterns = result['patterns']
            st.session_state.prediction = result['prediction']
            st.session_state.sentiment = result['sentiment']
            st.session_state.error = None
        except Exception as e:
            st.session_state.error = str(e)
            st.session_state.data = None

# Display results
if st.session_state.error:
    st.error(f"Error: {st.session_state.error}")
elif st.session_state.data:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Candlestick Chart")
        df = pd.DataFrame(st.session_state.data)
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
            title=f"{symbol} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Analysis Summary")
        st.markdown("**Predicted Next Day Close**")
        st.write(f"${st.session_state.prediction:.2f}")
        
        st.markdown("**News Sentiment**")
        sentiment_label = "Positive" if st.session_state.sentiment > 0 else "Negative" if st.session_state.sentiment < 0 else "Neutral"
        st.write(f"{sentiment_label} ({st.session_state.sentiment:.2f})")
        
        st.markdown("**Candlestick Patterns**")
        if st.session_state.patterns:
            pattern_df = pd.DataFrame(st.session_state.patterns)
            st.dataframe(pattern_df[['date', 'pattern', 'signal']], use_container_width=True)
        else:
            st.write("No patterns detected.")
    
    st.subheader("Technical Indicators")
    # Placeholder for additional visualizations (e.g., RSI, MACD)
    st.write("Technical indicators visualization coming soon.")

    # Download results
    if st.session_state.data:
        csv_data = [
            "Date,Open,High,Low,Close",
            *[f"{d['x']},{d['o']},{d['h']},{d['l']},{d['c']}" for d in st.session_state.data],
            "",
            "Date,Pattern,Signal",
            *[f"{p['date']},{p['pattern']},{p['signal']}" for p in st.session_state.patterns],
            "",
            "Prediction,Next Day Close",
            f",${st.session_state.prediction:.2f}",
            "",
            "Sentiment,Score",
            f",{st.session_state.sentiment:.2f}"
        ]
        csv = "\n".join(csv_data)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"{symbol}_analysis.csv",
            mime="text/csv"
        )
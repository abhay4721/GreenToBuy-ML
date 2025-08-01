import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import talib
import finnhub
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)
sia = SentimentIntensityAnalyzer()

def get_stock_data(symbol):
    data, _ = ts.get_daily(symbol=symbol, outputsize='full')
    data.columns = ['open', 'high', 'low', 'close', 'volume']
    data = data.sort_index()
    return data.tail(100)  # Last 100 days for robust training

def get_news_sentiment(symbol):
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    news = finnhub_client.company_news(symbol, _from=from_date, to=to_date)
    scores = [sia.polarity_scores(article['summary'])['compound'] for article in news if 'summary' in article]
    return np.mean(scores) if scores else 0.0

def detect_patterns(data):
    patterns = {}
    for pattern in ['CDLDOJI', 'CDLENGULFING', 'CDLHAMMER', 'CDLMORNINGSTAR']:
        values = getattr(talib, pattern)(data['open'], data['high'], data['low'], data['close'])
        patterns[pattern] = values
    detected = []
    for date, row in data.iterrows():
        for pattern, values in patterns.items():
            if values.loc[date] != 0:
                if pattern == 'CDLENGULFING':
                    signal = 'Bullish' if values.loc[date] > 0 else 'Bearish'
                else:
                    signal = 'Bullish' if pattern in ['CDLHAMMER', 'CDLMORNINGSTAR'] else 'Neutral'
                detected.append({
                    'date': date,
                    'pattern': pattern.replace('CDL', '').lower().capitalize(),
                    'signal': signal
                })
    return detected

def prepare_lstm_data(data, sentiment):
    data['sma_5'] = talib.SMA(data['close'], timeperiod=5)
    data['rsi'] = talib.RSI(data['close'], timeperiod=14)
    data['macd'], _, _ = talib.MACD(data['close'])
    data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
    data['sentiment'] = sentiment
    features = data[['close', 'sma_5', 'rsi', 'macd', 'atr', 'sentiment']].dropna()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    
    X, y = [], []
    lookback = 10
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0])  # Predict close price
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler, features.columns.tolist()

def train_lstm(X, y):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    return model

@app.get("/api/stock/{symbol}")
async def get_stock_analysis(symbol: str):
    try:
        # Fetch data
        data = get_stock_data(symbol)
        sentiment = get_news_sentiment(symbol)
        patterns = detect_patterns(data)
        
        # Prepare LSTM data
        X, y, scaler, feature_names = prepare_lstm_data(data, sentiment)
        if len(X) < 10:
            return JSONResponse(status_code=400, content={"error": "Insufficient data for prediction"})
        
        # Train LSTM
        model = train_lstm(X, y)
        
        # Predict next day's close
        last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
        pred_scaled = model.predict(last_sequence, verbose=0)
        pred_array = np.zeros((1, X.shape[2]))
        pred_array[0, 0] = pred_scaled[0, 0]
        prediction = scaler.inverse_transform(pred_array)[0, 0]
        
        # Format response
        chart_data = [
            {
                'x': date,
                'o': row['open'],
                'h': row['high'],
                'l': row['low'],
                'c': row['close']
            } for date, row in data.tail(30).iterrows()
        ]
        
        return {
            "data": chart_data,
            "patterns": patterns,
            "prediction": round(prediction, 2),
            "sentiment": round(sentiment, 2)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
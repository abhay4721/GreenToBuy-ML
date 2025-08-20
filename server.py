# server.py
import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import talib
import finnhub
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse
import sqlite3
import re

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)
sia = SentimentIntensityAnalyzer()

FOREX_TICKER_MAP = {
    "XAUUSD": "GC=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
}

def is_forex_symbol(symbol):
    forex_pattern = r'^[A-Z]{3}[A-Z]{3}$'
    return bool(re.match(forex_pattern, symbol))

def get_stock_data(symbol, interval='daily'):
    try:
        conn = sqlite3.connect('stock_data.db')
        table_name = f"{symbol}_{interval}"
        try:
            cached_data = pd.read_sql(f"SELECT * FROM {table_name}", conn, index_col='date', parse_dates=['date'])
            if not cached_data.empty and (datetime.now() - cached_data.index[-1]).days < 1:
                logger.debug(f"Using cached data for {symbol} ({interval})")
                conn.close()
                return cached_data
        except:
            pass
        logger.debug(f"Fetching new data for {symbol} ({interval})")
        
        interval_map = {
            '5min': ('7d', '5m'),
            '15min': ('60d', '15m'),
            '60min': ('60d', '60m'),
            'daily': ('1y', '1d')
        }
        period, yf_interval = interval_map.get(interval, ('1y', '1d'))
        
        yf_symbol = FOREX_TICKER_MAP.get(symbol, symbol) if is_forex_symbol(symbol) else symbol
        data = yf.download(yf_symbol, period=period, interval=yf_interval, progress=False)
        
        logger.debug(f"yfinance raw data for {yf_symbol} ({interval}): shape={data.shape}, columns={list(data.columns)}")
        
        if data.empty:
            logger.debug(f"yfinance returned no data for {yf_symbol}, trying Twelve Data")
            data = get_twelve_data(symbol, interval)
        
        if data.empty:
            logger.debug(f"Twelve Data returned no data, trying Alpha Vantage")
            data = get_alpha_vantage_data(symbol, interval)
        
        if data.empty:
            raise Exception("No data returned from sources")
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].lower().replace(' ', '_') for col in data.columns]
        else:
            data.columns = [str(col).lower().replace(' ', '_') for col in data.columns]
        
        logger.debug(f"Normalized columns for {symbol} ({interval}): {list(data.columns)}")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.debug(f"Missing columns {missing_columns}, trying alternative sources")
            data = get_twelve_data(symbol, interval) or get_alpha_vantage_data(symbol, interval)
            if data.empty:
                raise Exception(f"Missing required columns: {missing_columns}")
            data.columns = [str(col).lower().replace(' ', '_') for col in data.columns]
        
        data = data[required_columns]
        data.index = pd.to_datetime(data.index)
        data.index.name = 'date'
        data = data.sort_index()
        data.to_sql(table_name, conn, if_exists='replace')
        conn.close()
        return data.tail(100 if interval == 'daily' else 200)
    except Exception as e:
        logger.error(f"Error fetching data for {symbol} ({interval}): {str(e)}")
        raise Exception(f"Failed to fetch data: {str(e)}")

def get_twelve_data(symbol, interval):
    try:
        interval_map = {
            '5min': '5min',
            '15min': '15min',
            '60min': '1h',
            'daily': '1day'
        }
        td_interval = interval_map.get(interval, '1day')
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={td_interval}&apikey={TWELVE_DATA_KEY}"
        response = requests.get(url).json()
        
        if 'values' not in response or not response['values']:
            return pd.DataFrame()
        
        data = pd.DataFrame(response['values'])
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
        data = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
        data = data.rename_axis('date')
        return data.sort_index()
    except Exception as e:
        logger.error(f"Twelve Data error for {symbol} ({interval}): {str(e)}")
        return pd.DataFrame()

def get_alpha_vantage_data(symbol, interval):
    try:
        function_map = {
            '5min': 'TIME_SERIES_INTRADAY',
            '15min': 'TIME_SERIES_INTRADAY',
            '60min': 'TIME_SERIES_INTRADAY',
            'daily': 'TIME_SERIES_DAILY'
        }
        av_function = function_map.get(interval, 'TIME_SERIES_DAILY')
        av_interval = interval if 'min' in interval else None
        
        params = {
            'function': av_function,
            'symbol': symbol,
            'apikey': ALPHA_VANTAGE_KEY
        }
        if av_interval:
            params['interval'] = interval
        
        url = "https://www.alphavantage.co/query"
        response = requests.get(url, params=params).json()
        
        time_series_key = next((k for k in response if 'Time Series' in k), None)
        if not time_series_key:
            return pd.DataFrame()
        
        data = pd.DataFrame.from_dict(response[time_series_key], orient='index')
        data = data[['1. open', '2. high', '3. low', '4. close', '5. volume']].astype(float)
        data.columns = ['open', 'high', 'low', 'close', 'volume']
        data.index = pd.to_datetime(data.index)
        return data.sort_index()
    except Exception as e:
        logger.error(f"Alpha Vantage error for {symbol} ({interval}): {str(e)}")
        return pd.DataFrame()

def get_news_sentiment(symbol):
    try:
        scores = []
        # Try Finnhub first
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        news = finnhub_client.company_news(symbol, _from=from_date, to=to_date)
        for article in news:
            text = (article.get('headline', '') + ' ' + article.get('summary', '')).strip()
            if text and len(text) > 20:  # Ensure meaningful text
                score = sia.polarity_scores(text)['compound']
                scores.append(score)
        
        # If insufficient Finnhub data, use NewsAPI
        if len(scores) < 5:
            url = f"https://newsapi.org/v2/everything?q={symbol}&from={from_date}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
            response = requests.get(url).json()
            articles = response.get('articles', [])
            for article in articles:
                text = (article.get('title', '') + ' ' + article.get('description', '')).strip()
                if text and len(text) > 20:
                    score = sia.polarity_scores(text)['compound']
                    scores.append(score)
        
        sentiment = np.mean(scores) if scores else 0.0
        logger.debug(f"Sentiment for {symbol}: {sentiment} (from {len(scores)} articles)")
        return sentiment
    except Exception as e:
        logger.error(f"News sentiment error for {symbol}: {str(e)}")
        return 0.0

def detect_patterns(data):
    patterns = {}
    # Expanded list of candlestick patterns
    candlestick_patterns = [
        'CDLDOJI', 'CDLENGULFING', 'CDLHAMMER', 'CDLMORNINGSTAR',
        'CDLEVENINGSTAR', 'CDLHARAMI', 'CDLDRAGONFLYDOJI', 'CDLGRAVESTONEDOJI',
        'CDLPIERCING', 'CDLDARKCLOUDCOVER', 'CDLDOJISTAR', 'CDLMARUBOZU'
    ]
    for pattern in candlestick_patterns:
        try:
            values = getattr(talib, pattern)(data['open'], data['high'], data['low'], data['close'])
            patterns[pattern] = values
        except AttributeError as e:
            logger.error(f"TA-Lib error for {pattern}: {str(e)}")
            continue
    detected = []
    for i, (date, row) in enumerate(data.iterrows()):
        for pattern, values in patterns.items():
            if values.iloc[i] != 0:
                signal = (
                    'Bullish' if pattern in ['CDLHAMMER', 'CDLMORNINGSTAR', 'CDLPIERCING', 'CDLDRAGONFLYDOJI'] or
                    (pattern in ['CDLENGULFING', 'CDLHARAMI', 'CDLDOJISTAR', 'CDLMARUBOZU'] and values.iloc[i] > 0)
                    else 'Bearish' if pattern in ['CDLEVENINGSTAR', 'CDLDARKCLOUDCOVER', 'CDLGRAVESTONEDOJI'] or
                    (pattern in ['CDLENGULFING', 'CDLHARAMI', 'CDLDOJISTAR', 'CDLMARUBOZU'] and values.iloc[i] < 0)
                    else 'Neutral'
                )
                detected.append({
                    'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                    'pattern': pattern.replace('CDL', '').lower().capitalize(),
                    'signal': signal
                })
    return detected

def prepare_lstm_data(data, sentiment, interval='daily'):
    try:
        data['sma_5'] = talib.SMA(data['close'], timeperiod=5)
        data['rsi'] = talib.RSI(data['close'], timeperiod=14)
        data['macd'], _, _ = talib.MACD(data['close'])
        data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(data['close'], timeperiod=20)
        data['doji'] = talib.CDLDOJI(data['open'], data['high'], data['low'], data['close']) / 100
        data['engulfing'] = talib.CDLENGULFING(data['open'], data['high'], data['low'], data['close']) / 100
        data['hammer'] = talib.CDLHAMMER(data['open'], data['high'], data['low'], data['close']) / 100
        data['morningstar'] = talib.CDLMORNINGSTAR(data['open'], data['high'], data['low'], data['close']) / 100
        data['sentiment'] = sentiment
        features = data[['close', 'sma_5', 'rsi', 'macd', 'atr', 'bb_upper', 'bb_lower', 'doji', 'engulfing', 'hammer', 'morningstar', 'sentiment']].dropna()
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(features)
        
        lookback = {'5min': 120, '15min': 80, '60min': 60, 'daily': 10}.get(interval, 10)
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        
        return X, y, scaler, features.columns.tolist()
    except Exception as e:
        logger.error(f"Error preparing LSTM data for {interval}: {str(e)}")
        raise Exception(f"LSTM data preparation failed: {str(e)}")

def train_lstm(X, y):
    try:
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=50, batch_size=16, verbose=0)
        return model
    except Exception as e:
        logger.error(f"Error training LSTM: {str(e)}")
        raise Exception(f"LSTM training failed: {str(e)}")

def train_xgboost(X, y):
    try:
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X.reshape(X.shape[0], -1), y)
        return model
    except Exception as e:
        logger.error(f"Error training XGBoost: {str(e)}")
        raise Exception(f"XGBoost training failed: {str(e)}")

@app.get("/api/stock/{symbol}")
async def get_stock_analysis(symbol: str, model_type: str = "lstm"):
    try:
        logger.debug(f"Processing request for symbol: {symbol}, model: {model_type}")
        intervals = ['5min', '15min', '60min', 'daily']
        predictions = {}
        chart_data = {}
        patterns_dict = {}
        
        sentiment = get_news_sentiment(symbol)
        logger.debug(f"Sentiment: {sentiment}")
        
        latest_price = None
        for interval in intervals:
            try:
                data = get_stock_data(symbol, interval)
                logger.debug(f"Fetched data for {interval}: {data.shape}")
                if not data.empty and latest_price is None:
                    latest_price = {
                        'price': data['close'][-1],
                        'timestamp': data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1])
                    }
                patterns = detect_patterns(data)
                logger.debug(f"Patterns for {interval}: {len(patterns)} detected")
                patterns_dict[interval] = patterns
                
                X, y, scaler, feature_names = prepare_lstm_data(data, sentiment, interval)
                logger.debug(f"Data prepared for {interval}: X shape {X.shape}, y shape {y.shape}")
                if len(X) < 10:
                    predictions[interval] = {"error": "Insufficient data for prediction"}
                    continue
                
                model = train_lstm(X, y) if model_type.lower() == "lstm" else train_xgboost(X, y)
                logger.debug(f"Model ({model_type}) trained for {interval}")
                
                if model_type.lower() == "lstm":
                    last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
                    pred_scaled = model.predict(last_sequence, verbose=0)
                    pred_array = np.zeros((1, X.shape[2]))
                    pred_array[0, 0] = pred_scaled[0, 0]
                else:
                    last_sequence = X[-1].reshape(1, -1)
                    pred_scaled = model.predict(last_sequence)
                    pred_array = np.zeros((1, X.shape[2]))
                    pred_array[0, 0] = pred_scaled[0]
                
                prediction = scaler.inverse_transform(pred_array)[0, 0]
                predictions[interval] = round(prediction, 2)
                
                chart_data[interval] = [
                    {'x': date.isoformat() if hasattr(date, 'isoformat') else str(date), 'o': row['open'], 'h': row['high'], 'l': row['low'], 'c': row['close']}
                    for date, row in data.tail(30 if interval == 'daily' else 100).iterrows()
                ]
            except Exception as e:
                logger.error(f"Error processing {symbol} for {interval}: {str(e)}")
                predictions[interval] = {"error": str(e)}
                chart_data[interval] = []
                patterns_dict[interval] = []
        
        return {
            "chart_data": chart_data,
            "patterns": patterns_dict,
            "predictions": predictions,
            "sentiment": round(sentiment, 2),
            "is_forex": is_forex_symbol(symbol),
            "model_type": model_type,
            "latest_price": latest_price or {}
        }
    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
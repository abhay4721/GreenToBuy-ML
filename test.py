# test_yfinance.py
import yfinance as yf
import pandas as pd

symbol = "AAPL"
intervals = [('7d', '5m'), ('60d', '15m'), ('60d', '60m'), ('1y', '1d')]

for period, interval in intervals:
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    print(f"\nInterval: {interval}")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(data.head())
"""
data/fetcher.py

Purpose:
--------
Downloads or loads cached historical stock price data using AlphaVantage API.

Inputs:
-------
- symbol (str): Stock ticker (e.g., 'AAPL')
- interval (str): 'daily' or 'intraday'
- outputsize (str): 'full' or 'compact'

Outputs:
--------
- pd.DataFrame: Indexed price dataframe with OHLCV columns
"""

import os
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # Loads .env file with ALPHAVANTAGE_API_KEY


def fetch_stock_data(symbol: str, interval: str = 'daily', outputsize: str = 'full') -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)
    cache_file = f"data/{symbol}_{interval}.csv"

    if os.path.exists(cache_file):
        print("Found existing stock data")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df = df.sort_index()
        return df

    # === Fetch from AlphaVantage ===
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    ts = TimeSeries(key=api_key, output_format='pandas')

    if interval == 'daily':
        data, _ = ts.get_daily(symbol=symbol, outputsize=outputsize)
    elif interval == 'intraday':
        data, _ = ts.get_intraday(symbol=symbol, interval='60min', outputsize=outputsize)
    else:
        raise ValueError("Interval must be 'daily' or 'intraday'")

    df = data.rename(columns=lambda col: col.split('. ')[1])
    df = df.sort_index()

    # Save to CSV
    df.to_csv(cache_file)

    return df

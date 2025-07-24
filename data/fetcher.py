"""
data/fetcher.py

Purpose:
--------
Downloads historical stock price data using AlphaVantage API.

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
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    ts = TimeSeries(key=api_key, output_format='pandas')

    if interval == 'daily':
        data, _ = ts.get_daily(symbol=symbol, outputsize=outputsize)
    elif interval == 'intraday':
        data, _ = ts.get_intraday(symbol=symbol, interval='60min', outputsize=outputsize)
    else:
        raise ValueError("Interval must be 'daily' or 'intraday'")

    data = data.rename(columns=lambda col: col.split('. ')[1])
    data = data.sort_index()
    return data

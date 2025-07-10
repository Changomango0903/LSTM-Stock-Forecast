import os
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def fetch_stock_data(symbol: str, interval: str = 'daily', outputsize: str = 'full') -> pd.DataFrame:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    ts = TimeSeries(key=api_key, output_format='pandas')
    
    if interval == 'daily':
        data, _ = ts.get_daily(symbol=symbol, outputsize=outputsize)
    elif interval == 'intraday':
        data, _ = ts.get_intraday(symbol=symbol, interval='60min', outputsize=outputsize)
    else:
        raise ValueError("Invalid interval")
    
    data = data.rename(columns=lambda x: x.split('. ')[1])
    data = data.sort_index()
    return data
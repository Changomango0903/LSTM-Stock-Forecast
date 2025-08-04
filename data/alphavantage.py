"""
Fetch historical and latest daily prices from AlphaVantage.
Performs sanity checks on the JSON response.
"""

import logging
import requests
import pandas as pd
from config import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def fetch_historical(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Retrieves daily OHLCV for `symbol` between start and end (inclusive).
    Returns a DataFrame indexed by date, sorted ascending.
    Raises ValueError if no data or NaNs present.
    """
    logger.info(f"Fetching historical data for {symbol} from {start} to {end}")
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": CONFIG["av_api_key"],
        "outputsize": "full",
        "datatype": "json",
    }
    resp = requests.get(CONFIG["av_base_url"], params=params, timeout=30)
    resp.raise_for_status()
    raw = resp.json().get("Time Series (Daily)")
    if not raw:
        raise ValueError("AlphaVantage returned no 'Time Series (Daily)' data")
    df = (
        pd.DataFrame.from_dict(raw, orient="index")
          .rename(columns={
               "1. open":"open","2. high":"high","3. low":"low",
               "4. close":"close","5. volume":"volume"})
          .astype(float)
    )
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().loc[start:end]
    if df.empty or df.isna().any().any():
        raise ValueError("Fetched historical data is empty or contains NaNs")
    logger.debug(f"Fetched {len(df)} rows")
    return df


def fetch_latest_open(symbol: str) -> float:
    """
    Retrieves the most recent trading day's open price for `symbol`.
    Raises ValueError if no data found.
    """
    logger.info(f"Fetching latest open for {symbol}")
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": CONFIG["av_api_key"],
        "outputsize": "compact",
        "datatype": "json",
    }
    resp = requests.get(CONFIG["av_base_url"], params=params, timeout=10)
    resp.raise_for_status()
    ts = resp.json().get("Time Series (Daily)")
    if not ts:
        raise ValueError("AlphaVantage returned no latest series")
    date, data = sorted(ts.items(), reverse=True)[0]
    open_price = float(data["1. open"])
    logger.debug(f"Latest open on {date}: {open_price}")
    return open_price

"""
Compute and lag technical indicators to avoid leakage:
- SMA_20, SMA_50
- RSI_14
- Volatility (20-day std of returns)

Each indicator at date t uses data through t-1 only.
"""

import pandas as pd

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
      df: DataFrame with 'close' and other OHLCV columns.
    Returns:
      df with new columns ['SMA_20','SMA_50','RSI_14','Volatility'],
      each shifted by 1 day, then initial NaNs dropped.
    """
    out = df.copy()

    out["SMA_20_raw"] = out["close"].rolling(20).mean()
    out["SMA_50_raw"] = out["close"].rolling(50).mean()

    delta = out["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up   = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    out["RSI_14_raw"] = 100 - (100 / (1 + (ma_up / ma_down)))

    out["Volatility_raw"] = out["close"].pct_change().rolling(20).std()

    # Shift each raw indicator so day-t uses only up to t-1
    for col in ["SMA_20_raw","SMA_50_raw","RSI_14_raw","Volatility_raw"]:
        new = col.replace("_raw","")
        out[new] = out[col].shift(1)
        out.drop(columns=[col], inplace=True)

    out = out.dropna()
    if out.isna().any().any():
        raise ValueError("NaNs remain after indicator generation")
    return out

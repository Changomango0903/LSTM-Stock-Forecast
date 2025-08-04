"""
Compute and lag technical indicators to avoid leakage:
- SMA_20, SMA_50
- RSI_14
- Volatility (20-day std of returns)
- Return_1 (1-day pct change)
- Return_5 (5-day pct change)
- Volume_Change (1-day pct change in volume)
- Return_Skew (20-day skew of returns)

Each indicator at date t uses data through t-1 only.
"""

import pandas as pd

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
      df: DataFrame with 'close' and 'volume'.
    Returns:
      df with new columns [
        'SMA_20','SMA_50','RSI_14','Volatility',
        'Return_1','Return_5','Volume_Change','Return_Skew'
      ], each shifted by 1 day, then initial NaNs dropped.
    """
    out = df.copy()

    # Moving averages
    out["SMA_20_raw"] = out["close"].rolling(20).mean()
    out["SMA_50_raw"] = out["close"].rolling(50).mean()

    # RSI
    delta = out["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up   = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    out["RSI_14_raw"] = 100 - (100 / (1 + (ma_up / ma_down)))

    # Volatility: rolling std of returns
    out["Volatility_raw"] = out["close"].pct_change().rolling(20).std()

    # New momentum features
    out["Return_1_raw"]        = out["close"].pct_change(1)
    out["Return_5_raw"]        = out["close"].pct_change(5)
    out["Volume_Change_raw"]   = out["volume"].pct_change(1)
    # Skew of 20-day returns
    out["Return_Skew_raw"]      = out["close"].pct_change().rolling(20).skew()

    # Shift each raw indicator so day-t uses only up to t-1
    raw_cols = [
        "SMA_20_raw","SMA_50_raw","RSI_14_raw","Volatility_raw",
        "Return_1_raw","Return_5_raw","Volume_Change_raw","Return_Skew_raw"
    ]
    for col in raw_cols:
        new = col.replace("_raw","")
        out[new] = out[col].shift(1)
        out.drop(columns=[col], inplace=True)

    # Drop any rows with NaNs from warm-up or shifting
    out = out.dropna()
    if out.isna().any().any():
        raise ValueError("NaNs remain after indicator generation")
    return out

"""
features/indicators.py

Purpose:
--------
Adds a wide set of technical indicators to the stock price dataframe for use
in feature engineering.

Inputs:
-------
- df (pd.DataFrame): DataFrame containing ['open', 'high', 'low', 'close', 'volume']

Outputs:
--------
- pd.DataFrame: Original df with added technical features.
"""

import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators like SMA, EMA, RSI, MACD, Bollinger Bands, OBV, Ichimoku.

    Returns:
        pd.DataFrame: DataFrame with added columns.
    """

    # === Basic Returns ===
    df['Returns'] = df['close'].pct_change()

    # === Moving Averages ===
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # === Bollinger Bands ===
    std = df['close'].rolling(window=20).std()
    df['BB_Middle'] = df['SMA_20']
    df['BB_Upper'] = df['BB_Middle'] + 2 * std
    df['BB_Lower'] = df['BB_Middle'] - 2 * std

    # === RSI (Relative Strength Index) ===
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(14).mean()
    avg_loss = down.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # === MACD (Moving Average Convergence Divergence) ===
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # === Momentum ===
    df['Momentum'] = df['close'] - df['close'].shift(10)

    # === On-Balance Volume (OBV) ===
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i - 1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # === Ichimoku Components ===
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2

    df['Senkou_span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['Senkou_span_B'] = ((high_52 + low_52) / 2).shift(26)

    df['Chikou_span'] = df['close'].shift(-26)

    # === Rolling Volatility ===
    df['Volatility'] = df['Returns'].rolling(window=20).std()

    df.dropna(inplace=True)
    return df

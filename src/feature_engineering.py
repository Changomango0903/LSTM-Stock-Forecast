import pandas as pd
import numpy as np

def train_test_split(df: pd.DataFrame, sequence_length: int = 60, test_size: float = 0.2):
    data = df[['close', 'Returns', 'SMA_20', 'SMA_50', 'Volatility']].values
    sequences, targets = [], []
    
    for i in range(sequence_length, len(data)):
        sequences.append(data[i-sequence_length:i])
        targets.append(data[i][0])  # Predict closing price

    split = int(len(sequences) * (1 - test_size))
    X_train, X_test = np.array(sequences[:split]), np.array(sequences[split:])
    y_train, y_test = np.array(targets[:split]), np.array(targets[split:])
    return X_train, X_test, y_train, y_test

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds multiple technical indicators to the stock price dataframe.
    Args:
        df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
    Returns:
        pd.DataFrame: DataFrame with added technical features.
    """

    # Daily returns
    df['Returns'] = df['close'].pct_change()

    # Simple Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()

    # Exponential Moving Average
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Bollinger Bands
    df['BB_Middle'] = df['SMA_20']
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['close'].rolling(window=20).std()

    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / roll_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + RS))

    # Moving Average Convergence Divergence (MACD)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Momentum
    df['Momentum'] = df['close'] - df['close'].shift(10)

    # On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i - 1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # Ichimoku Cloud Components
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

    # Rolling Volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()

    df.dropna(inplace=True)

    return df

"""
data/split.py

Purpose:
--------
Produces time-aligned rolling sequences for time-series forecasting.
Returns matching date indices for test set to support proper plotting.

Inputs:
-------
- df (pd.DataFrame): Must include features + datetime index
- sequence_length (int): Lookback window size
- test_size (float): Fraction of samples to reserve for test

Outputs:
--------
- X_train, X_test, y_train, y_test: np.ndarray
- train_dates, test_dates: np.ndarray of datetime indices
"""

import numpy as np
import pandas as pd


def create_sequences(df: pd.DataFrame, sequence_length: int = 60, test_size: float = 0.2):
    feature_cols = ['close', 'Returns', 'SMA_20', 'SMA_50', 'Volatility']
    features = df[feature_cols].values
    dates = df.index.values

    sequences, targets, target_dates = [], [], []

    for i in range(sequence_length, len(df)):
        sequences.append(features[i - sequence_length:i])
        targets.append(features[i][0])         # closing price
        target_dates.append(dates[i])          # date of the target price

    X = np.array(sequences)
    y = np.array(targets)
    d = np.array(target_dates)

    split = int(len(X) * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:], d[:split], d[split:]

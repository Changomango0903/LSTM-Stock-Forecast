"""
Convert a cleaned DataFrame into overlapping sequences for LSTM.
Validates that df is long enough.
"""

import numpy as np

def create_sequences(df, sequence_length: int):
    """
    Args:
      df: DataFrame with 'close' in its columns (and all features computed & causal).
      sequence_length: number of days per sample.
    Returns:
      X: np.array shape (n_samples, sequence_length, n_features)
      y: np.array shape (n_samples,)
      dates: np.array of the target dates
    """
    if len(df) <= sequence_length:
        raise ValueError("Not enough data to create sequences")
    feats = df.values
    dates = df.index.to_numpy()
    X, y, d = [], [], []
    close_idx = df.columns.get_loc("close")
    for i in range(sequence_length, len(df)):
        X.append(feats[i-sequence_length:i])
        y.append(feats[i, close_idx])
        d.append(dates[i])
    return np.array(X), np.array(y), np.array(d)

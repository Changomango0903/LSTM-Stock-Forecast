"""
Modular, strictly causal denoising:
- raw: no change
- ema: Exponential Moving Average (adjust=False)
- kalman: forward Kalman filter only
"""

import pandas as pd
from pykalman import KalmanFilter

def apply_denoising(series: pd.Series, method: str = "kalman") -> pd.Series:
    """
    Returns a denoised version of `series`. Raises on unknown method.
    """
    if method == "raw":
        return series.copy()
    if method == "ema":
        # adjust=False => uses only past data
        return series.ewm(span=20, adjust=False).mean()
    if method == "kalman":
        # forward filter: no smoothing pass
        kf = KalmanFilter(initial_state_mean=series.iloc[0], n_dim_obs=1)
        state_means, _ = kf.filter(series.values)
        return pd.Series(state_means.flatten(), index=series.index)
    raise ValueError(f"Unknown denoising method: {method}")

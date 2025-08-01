"""
Kalman filter smoothing using pykalman (causal version — avoids future leakage).
"""

import pandas as pd
import numpy as np
from pykalman import KalmanFilter


def apply_kalman(series):
    """
    Applies Kalman filter (causal — no future data used).

    Args:
        series (pd.Series): Time series to be filtered

    Returns:
        pd.Series: Filtered output with same index
    """
    kf = KalmanFilter(
        initial_state_mean=series.iloc[0],
        n_dim_obs=1,
        transition_matrices=[1],
        observation_matrices=[1],
        observation_covariance=1.0,
        transition_covariance=0.01
    )

    filtered_state_means, _ = kf.filter(series.values)
    return pd.Series(filtered_state_means.flatten(), index=series.index)

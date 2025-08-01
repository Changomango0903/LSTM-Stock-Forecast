"""
Kalman filter smoothing using pykalman.
"""

import pandas as pd
from pykalman import KalmanFilter
import numpy as np


def apply_kalman(series):
    """
    Applies 1D Kalman smoothing.

    Args:
        series (pd.Series)

    Returns:
        pd.Series
    """
    kf = KalmanFilter(initial_state_mean=series.values[0],
                      n_dim_obs=1,
                      transition_matrices=[1],
                      observation_matrices=[1],
                      observation_covariance=1.0,
                      transition_covariance=0.01)

    smoothed_state, _ = kf.smooth(series.values)
    return pd.Series(smoothed_state.flatten(), index=series.index)

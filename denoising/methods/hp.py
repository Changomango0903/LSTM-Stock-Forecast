"""
Hodrick–Prescott Filter (Trend-Cycle Decomposition)
"""

import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter


def apply_hp(series, lamb=1600):
    """
    Applies HP filter to decompose signal and return the smoothed trend.

    Args:
        series (pd.Series): Time series
        lamb (float): Smoothing parameter (1600 = monthly, 100000 = daily)

    Returns:
        pd.Series: Smoothed trend component
    """
    trend, _ = hpfilter(series, lamb=lamb)
    return trend

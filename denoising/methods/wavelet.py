"""
Wavelet transform smoothing using pywt.
"""

import pywt
import pandas as pd
import numpy as np


def apply_wavelet(series, wavelet='db4', level=1):
    """
    Applies wavelet denoising using soft thresholding on high-frequency coefficients.

    Args:
        series (pd.Series)
        wavelet (str): Type of wavelet
        level (int): Decomposition level

    Returns:
        pd.Series
    """
    coeffs = pywt.wavedec(series.values, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(series)))
    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
    smoothed = pywt.waverec(coeffs, wavelet)
    return pd.Series(smoothed[:len(series)], index=series.index)

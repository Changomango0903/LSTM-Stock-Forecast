"""
denoising/denoise.py

Purpose:
--------
Central router for all denoising methods.
"""

from denoising.methods.ema import apply_ema
from denoising.methods.fft import apply_fft


def apply_denoising(series, method='raw'):
    """
    Apply one of the supported denoising methods to a price series.

    Args:
        series (pd.Series): Price or return series
        method (str): One of ['raw', 'ema', 'fft', 'hp', 'kalman', 'wavelet']

    Returns:
        pd.Series: Denoised version
    """
    if method == 'raw':
        return series
    elif method == 'ema':
        return apply_ema(series)
    elif method == 'fft':
        return apply_fft(series)
    elif method == 'hp':
        return apply_hp(series)
    elif method == 'kalman':
        return apply_kalman(series)
    elif method == 'wavelet':
        return apply_wavelet(series)
    else:
        raise ValueError(f"Unsupported denoising method: {method}")

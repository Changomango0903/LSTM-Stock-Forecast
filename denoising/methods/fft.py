"""
Low-pass filtering via FFT (removes high frequency noise).
"""

import numpy as np
import pandas as pd

def apply_fft(series, cutoff_ratio=0.1):
    fft_vals = np.fft.fft(series.values)
    fft_freqs = np.fft.fftfreq(len(fft_vals))

    cutoff = cutoff_ratio * np.max(np.abs(fft_freqs))
    filtered_fft = fft_vals.copy()
    filtered_fft[np.abs(fft_freqs) > cutoff] = 0

    smoothed = np.fft.ifft(filtered_fft)
    return pd.Series(np.real(smoothed), index=series.index)

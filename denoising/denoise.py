def apply_denoising(series: pd.Series, method: str = 'raw') -> pd.Series:
    """
    Apply one of the supported denoising methods to a time-series.

    Parameters:
        series (pd.Series): Raw close or return series
        method (str): One of ['raw', 'ema', 'fft', 'hp', 'wavelet', 'kalman']

    Returns:
        pd.Series: Denoised output with same index
    """

"""
Exponential Moving Average smoothing.
"""

def apply_ema(series, span=10):
    return series.ewm(span=span, adjust=False).mean()

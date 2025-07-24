"""
training/metrics.py

Purpose:
--------
Calculate financial metrics from predictions, for evaluation and logging.

Inputs:
-------
- preds (np.ndarray): Predicted prices
- actuals (np.ndarray): Actual prices

Outputs:
--------
- directional_acc (float)
- win_rate (float)
- mean_signal_return (float)
"""

import numpy as np


def compute_finance_metrics(preds, actuals):
    preds_shift = np.roll(preds, 1)
    preds_shift[0] = preds_shift[1]  # first value edge fix

    actual_returns = np.diff(actuals) / actuals[:-1]
    predicted_direction = np.sign(np.diff(preds_shift))
    actual_direction = np.sign(actual_returns)

    correct = predicted_direction == actual_direction
    directional_acc = np.mean(correct)
    win_rate = directional_acc  # long-only strategy

    mean_signal_return = np.mean(actual_returns * predicted_direction)

    return directional_acc, win_rate, mean_signal_return

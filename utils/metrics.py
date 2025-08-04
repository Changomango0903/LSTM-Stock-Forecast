"""
Compute common financial metrics:
- Directional Accuracy
- RMSE
"""

import numpy as np

def directional_accuracy(preds, actuals):
    return np.mean(np.sign(preds[1:]-preds[:-1]) == np.sign(actuals[1:]-actuals[:-1]))

def rmse(preds, actuals):
    return np.sqrt(np.mean((preds-actuals)**2))

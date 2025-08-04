"""
Holdout-only evaluation (no retraining).
"""

import logging
from training.trainer import train_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def evaluate_holdout(model, X_hold, y_hold, y_scaler, dates):
    """
    Computes RMSE & DA on holdout via a single pass of train_model.
    Returns {"rmse":…, "da":…}.
    """
    _, preds, actuals, _ = train_model(model, None, None, X_hold, y_hold, y_scaler, dates)
    rmse = ((preds-actuals)**2).mean()**0.5
    da   = ((preds[1:]-preds[:-1])*(actuals[1:]-actuals[:-1])>0).mean()
    logger.info(f"Holdout RMSE={rmse:.4f}, DA={da:.4f}")
    return {"rmse":rmse, "da":da}

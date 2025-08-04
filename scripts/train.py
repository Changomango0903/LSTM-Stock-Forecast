"""
Full retrain + holdout evaluation with training curves.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import logging, wandb
from config import CONFIG
from data.alphavantage import fetch_historical
from denoising.denoise import apply_denoising
from features.indicators import add_technical_indicators
from data.split import create_sequences
from data.scaler import scale_data
from training.trainer import train_model
from utils.plots import plot_training_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    wandb.init(project=CONFIG["wandb_project"], name="full_retrain", config=CONFIG)
    df = fetch_historical(CONFIG["default_symbol"], CONFIG["start_date"], CONFIG["end_date"])
    df["close"] = apply_denoising(df["close"], CONFIG["denoising_method"])
    df = add_technical_indicators(df)
    df = df[CONFIG["feature_cols"]]

    X, y, dates = create_sequences(df, CONFIG["sequence_length"])
    split = len(X) - CONFIG["holdout_size"]
    X_tr, X_hd = X[:split], X[split:]
    y_tr, y_hd = y[:split], y[split:]
    X_tr_s, X_hd_s, y_tr_s, y_hd_s, _, y_sc = scale_data(X_tr, X_hd, y_tr, y_hd)

    best_rmse, preds, actuals, hd_dates, history = train_model(
        None, X_tr_s, y_tr_s, X_hd_s, y_hd_s, y_sc, dates[split:], return_history=True
    )
    logger.info(f"Best holdout RMSE: {best_rmse:.4f}")
    wandb.finish()

    # Plot training curves
    epochs = list(range(1, CONFIG["epochs"]+1))
    plot_training_curves(
        epochs,
        history["train_losses"],
        history["holdout_rmses"],
        history["holdout_das"]
    )

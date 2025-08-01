"""
utils/plots.py

Purpose:
--------
Handles all matplotlib visualizations used during training and evaluation.

Saves plots to a dedicated 'plots/' folder and logs them to Weights & Biases.

Inputs:
-------
- Predictions, targets (np.ndarray)
- Epoch-wise loss history (dict)
- Corresponding datetime indices (optional)

Outputs:
--------
- PNG plots in /plots
- wandb image logs
"""

import matplotlib.pyplot as plt
import numpy as np
import wandb
import os


def plot_predictions(preds, actuals, model_name: str, dates=None):
    """
    Plot predicted vs actual time series.

    Args:
        preds (np.ndarray): Predicted values
        actuals (np.ndarray): Ground truth
        model_name (str): e.g., 'LSTM'
        dates (np.ndarray, optional): Corresponding datetime index
    """
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(12, 6))
    if dates is not None:
        plt.plot(dates, actuals, label="Actual")
        plt.plot(dates, preds, label="Predicted")
    else:
        plt.plot(actuals, label="Actual")
        plt.plot(preds, label="Predicted")

    plt.title(f"{model_name} Predictions vs Actuals")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    file = os.path.join("plots", f"{model_name}_predictions.png")
    plt.savefig(file)
    wandb.log({f"{model_name}_Predictions_vs_Actuals": wandb.Image(file)})
    plt.close()


def plot_loss_curve(history, model_name: str):
    """
    Plot training/validation loss curve.

    Args:
        history (dict): Contains 'train_loss' and 'val_loss'
        model_name (str)
    """
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label="Train Loss")
    plt.plot(history['val_loss'], label="Val Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()

    file = os.path.join("plots", f"{model_name}_loss_curve.png")
    plt.savefig(file)
    wandb.log({f"{model_name}_Loss_Curve": wandb.Image(file)})
    plt.close()


def plot_combined(actuals, lstm_preds, attn_preds, dates, title=None):
    """
    Plot combined forecast from LSTM and Attention LSTM vs actuals.

    Args:
        actuals (np.ndarray): Ground truth
        lstm_preds (np.ndarray): LSTM predictions
        attn_preds (np.ndarray): Attention LSTM predictions
        dates (np.ndarray, optional): Corresponding datetime index
    """
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(14, 7))
    if dates is not None:
        plt.plot(dates, actuals, label="Actual", color="black", linewidth=2)
        plt.plot(dates, lstm_preds, label="LSTM Predicted", linestyle="--")
        plt.plot(dates, attn_preds, label="Attention LSTM Predicted", linestyle="--")
    else:
        plt.plot(actuals, label="Actual", color="black", linewidth=2)
        plt.plot(lstm_preds, label="LSTM Predicted", linestyle="--")
        plt.plot(attn_preds, label="Attention LSTM Predicted", linestyle="--")

    if title:
        plt.title(title)
    else:    
        plt.title("Combined Forecast: LSTM vs Attention LSTM vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    file = os.path.join("plots", "combined_lstm_attention_predictions.png")
    plt.savefig(file)
    wandb.log({"Combined_LSTM_vs_Attention_vs_Actuals": wandb.Image(file)})
    plt.close()

"""
evaluate_holdout.py

Purpose:
--------
Evaluate trained models (LSTM, AttentionLSTM) on strictly held-out, unseen future data.

Workflow:
---------
- Load historical data
- Split into train and holdout (chronological)
- Train model on training set
- Evaluate predictions on holdout set
- Log metrics and comparison plots to wandb

CLI Example:
------------
$ python evaluate_holdout.py --symbol AAPL --model lstm --holdout_size 250
$ python evaluate_holdout.py --symbol TSLA --model attn --denoise_method kalman
"""

import argparse
import wandb
import pandas as pd
from config import CONFIG
from data.fetcher import fetch_stock_data
from data.scaler import scale_data
from data.split import create_sequences
from features.indicators import add_technical_indicators
from models.lstm import LSTMModel
from models.attention_lstm import AttentionLSTM
from training.trainer import train_model
from training.metrics import compute_finance_metrics
from utils.plots import plot_combined
from denoising.denoise import apply_denoising

import matplotlib.pyplot as plt


def evaluate_holdout(symbol: str, model_type: str, holdout_size: int, denoise_method: str):
    """
    Evaluate a model on a future holdout segment.

    Args:
        symbol (str): Stock ticker
        model_type (str): 'lstm' or 'attn'
        holdout_size (int): Number of data points in holdout
        denoise_method (str): Denoising method to apply
    Returns:
        None
    """
    wandb.init(project=CONFIG['project_name'], name=f"Holdout_{symbol}_{model_type}_{denoise_method}", config=CONFIG)

    df = fetch_stock_data(symbol)
    df = df.loc[CONFIG['start_date']:CONFIG['end_date']]
    df = add_technical_indicators(df)

    if denoise_method != 'raw':
        raw_close = df['close'].copy()
        df['close'] = apply_denoising(raw_close, method=denoise_method)
        plt.figure(figsize=(12, 5))
        plt.plot(raw_close, label='Raw Close')
        plt.plot(df['close'], label=f'{denoise_method} Smoothed')
        plt.legend(); plt.tight_layout()
        wandb.log({f"{symbol}_Raw_vs_Smoothed_{denoise_method}": wandb.Image(plt)})
        plt.close()

    total_len = len(df)
    train_df = df.iloc[:total_len - holdout_size]
    holdout_df = df.iloc[total_len - holdout_size:]

    # Sequence & scale
    X_train, _, y_train, _, _, _ = create_sequences(train_df, sequence_length=CONFIG['sequence_length'])
    _, X_holdout, _, y_holdout, _, holdout_dates = create_sequences(df, sequence_length=CONFIG['sequence_length'])
    X_train, X_holdout, y_train, y_holdout, X_scaler, y_scaler = scale_data(X_train, X_holdout, y_train, y_holdout)

    # Select model
    if model_type == 'lstm':
        model = LSTMModel(input_size=X_train.shape[2], **CONFIG)
        model_name = 'LSTM'
    else:
        model = AttentionLSTM(input_size=X_train.shape[2], **CONFIG)
        model_name = 'Attention_LSTM'

    # Train
    rmse, preds, actuals, _ = train_model(model, X_train, y_train, X_holdout, y_holdout,
                                          CONFIG | {'model_name': model_name}, y_scaler,
                                          test_dates=holdout_dates)

    # Metrics
    da, wr, msr = compute_finance_metrics(preds, actuals)
    wandb.log({
        f"{model_name}_Holdout_RMSE": rmse,
        f"{model_name}_Holdout_Directional_Accuracy": da,
        f"{model_name}_Holdout_Win_Rate": wr,
        f"{model_name}_Holdout_Mean_Signal_Return": msr,
    })

    # Plot
    plot_combined(actuals, preds, preds, dates=holdout_dates, title=f"{symbol} Holdout - {model_name}")
    print(holdout_dates[0], "→", holdout_dates[-1])
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['lstm', 'attn'], default='lstm')
    parser.add_argument('--holdout_size', type=int, default=250)
    parser.add_argument('--denoise_method', type=str, default='raw')
    args = parser.parse_args()

    evaluate_holdout(args.symbol, args.model, args.holdout_size, args.denoise_method)

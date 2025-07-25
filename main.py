"""
main.py

Purpose:
--------
Unified entry point to run training or evaluation via CLI flags.

Supports:
    --mode: 'train' or 'evaluate'
    --model: 'lstm', 'attn', or 'baseline'
    --symbol: Stock ticker symbol

Example:
--------
$ python main.py --mode train --model lstm --symbol AAPL
$ python main.py --mode evaluate --symbol AAPL

Dependencies:
-------------
- Requires proper AlphaVantage API key set via `.env`
- Uses wandb for experiment tracking

Returns:
--------
None (results are logged and plotted)
"""

import argparse
from config import CONFIG
from training.trainer import train_model
from training.metrics import compute_finance_metrics
from data.fetcher import fetch_stock_data
from data.split import create_sequences
from data.scaler import scale_data
from features.indicators import add_technical_indicators
from models.lstm import LSTMModel
from models.attention_lstm import AttentionLSTM
from models.baselines import train_baseline_models
from utils.plots import plot_predictions, plot_loss_curve, plot_combined

import wandb
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True)
    parser.add_argument('--model', choices=['lstm', 'attn', 'baseline'], default='lstm')
    parser.add_argument('--symbol', type=str, default=CONFIG['default_symbol'])
    args = parser.parse_args()

    wandb.init(project=CONFIG['project_name'], config=CONFIG)

    # === Load & preprocess ===
    df = fetch_stock_data(args.symbol)
    df = df.loc[CONFIG['start_date']:CONFIG['end_date']]
    print(df)
    print(df.shape)
    df = add_technical_indicators(df)
    X_train, X_test, y_train, y_test, train_dates, test_dates = create_sequences(df, sequence_length=CONFIG['sequence_length'])
    X_train, X_test, y_train, y_test, X_scaler, y_scaler = scale_data(X_train, X_test, y_train, y_test)

    if args.mode == 'train':
        if args.model == 'lstm':
            model = LSTMModel(input_size=X_train.shape[2], **CONFIG)
            name = 'LSTM'
        elif args.model == 'attn':
            model = AttentionLSTM(input_size=X_train.shape[2], **CONFIG)
            name = 'Attention_LSTM'
        else:
            raise ValueError("Baseline training not supported in 'train' mode. Use 'evaluate'.")

        rmse, preds, actuals, history = train_model(model, X_train, y_train, X_test, y_test,
                                                    CONFIG | {'model_name': name},
                                                    y_scaler,
                                                    test_dates=test_dates)

        plot_predictions(preds, actuals, name)
        plot_loss_curve(history, name)

    elif args.mode == 'evaluate':
        # === Baselines ===
        baseline_rmse, _ = train_baseline_models(X_train, y_train, X_test, y_test, test_dates)

        # === LSTM ===
        lstm = LSTMModel(input_size=X_train.shape[2], **CONFIG)
        lstm_rmse, lstm_preds, lstm_actuals, lstm_hist = train_model(
            lstm, X_train, y_train, X_test, y_test,
            CONFIG | {'model_name': 'LSTM'}, y_scaler
        )

        # === Attention LSTM ===
        attn = AttentionLSTM(input_size=X_train.shape[2], **CONFIG)
        attn_rmse, attn_preds, attn_actuals, attn_hist = train_model(
            attn, X_train, y_train, X_test, y_test,
            CONFIG | {'model_name': 'Attention_LSTM'}, y_scaler
        )

        # === Financial Metrics ===
        lstm_metrics = compute_finance_metrics(lstm_preds, lstm_actuals)
        attn_metrics = compute_finance_metrics(attn_preds, attn_actuals)

        wandb.log({
            "LSTM_Directional_Accuracy": lstm_metrics[0],
            "LSTM_Win_Rate": lstm_metrics[1],
            "LSTM_Mean_Signal_Return": lstm_metrics[2],
            "AttnLSTM_Directional_Accuracy": attn_metrics[0],
            "AttnLSTM_Win_Rate": attn_metrics[1],
            "AttnLSTM_Mean_Signal_Return": attn_metrics[2],
        })

        plot_combined(lstm_actuals, lstm_preds, attn_preds, dates=test_dates)

    wandb.finish()


if __name__ == "__main__":
    main()

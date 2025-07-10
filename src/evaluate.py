import wandb
import matplotlib.pyplot as plt

from data_loader import fetch_stock_data
from feature_engineering import add_technical_indicators, train_test_split
from baseline_models import train_baseline_models
from lstm_models import LSTMModel, AttentionLSTM
from train import train_model, scale_data

import numpy as np


def compute_finance_metrics(preds, actuals):
    """
    Calculate directional accuracy, win rate, mean signal return.
    """
    preds_shift = np.roll(preds, 1)  # yesterday's prediction for today's signal
    preds_shift[0] = preds_shift[1]  # edge fix

    actual_returns = np.diff(actuals) / actuals[:-1]
    pred_direction = np.sign(np.diff(preds_shift))
    true_direction = np.sign(actual_returns)

    correct = pred_direction == true_direction
    directional_acc = np.mean(correct)
    win_rate = directional_acc  # same for long-only

    mean_return = np.mean(actual_returns * pred_direction)

    return directional_acc, win_rate, mean_return


CONFIG = {
    'sequence_length': 60,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 20,
    'project_name': 'stock_forecasting_pipeline'
}


def compare_all_models(symbol="AAPL"):
    wandb.init(project=CONFIG['project_name'], config=CONFIG)

    # === Load & engineer ===
    df = fetch_stock_data(symbol)
    df = add_technical_indicators(df)
    X_train, X_test, y_train, y_test = train_test_split(df, sequence_length=CONFIG['sequence_length'])

    # === Scale ===
    X_train, X_test, y_train, y_test, X_scaler, y_scaler = scale_data(X_train, X_test, y_train, y_test)

    # === Baselines ===
    print(f"\n🔍 Training sklearn baselines for {symbol}...")
    baseline_rmse, _ = train_baseline_models(X_train, y_train, X_test, y_test)
    for name, rmse in baseline_rmse.items():
        print(f"✅ {name} RMSE: {rmse:.4f}")

    # === LSTM ===
    print("\n🔍 Training LSTM...")
    lstm = LSTMModel(
        input_size=X_train.shape[2],
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    )
    lstm_rmse, lstm_preds, lstm_actuals, _ = train_model(
        lstm, X_train, y_train, X_test, y_test,
        CONFIG | {'model_name': 'LSTM'},
        y_scaler
    )
    print(f"✅ LSTM RMSE: {lstm_rmse:.4f}")

    # === Attention LSTM ===
    print("\n🔍 Training Attention LSTM...")
    attn_lstm = AttentionLSTM(
        input_size=X_train.shape[2],
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    )
    attn_rmse, attn_preds, attn_actuals, _ = train_model(
        attn_lstm, X_train, y_train, X_test, y_test,
        CONFIG | {'model_name': 'Attention_LSTM'},
        y_scaler
    )
    print(f"✅ Attention LSTM RMSE: {attn_rmse:.4f}")

    # === Financial metrics ===
    lstm_dir_acc, lstm_win_rate, lstm_mean_return = compute_finance_metrics(lstm_preds, lstm_actuals)
    attn_dir_acc, attn_win_rate, attn_mean_return = compute_finance_metrics(attn_preds, attn_actuals)

    wandb.log({
        "LSTM_Directional_Accuracy": lstm_dir_acc,
        "LSTM_Win_Rate": lstm_win_rate,
        "LSTM_Mean_Signal_Return": lstm_mean_return,
        "AttnLSTM_Directional_Accuracy": attn_dir_acc,
        "AttnLSTM_Win_Rate": attn_win_rate,
        "AttnLSTM_Mean_Signal_Return": attn_mean_return,
    })

    print("\n📈 Financial metrics:")
    print(f"LSTM Directional Accuracy: {lstm_dir_acc:.4f}")
    print(f"LSTM Win Rate: {lstm_win_rate:.4f}")
    print(f"LSTM Mean Signal Return: {lstm_mean_return:.6f}")
    print(f"AttnLSTM Directional Accuracy: {attn_dir_acc:.4f}")
    print(f"AttnLSTM Win Rate: {attn_win_rate:.4f}")
    print(f"AttnLSTM Mean Signal Return: {attn_mean_return:.6f}")

    # === Combined plot ===
    plt.figure(figsize=(14, 7))
    plt.plot(lstm_actuals, label="Actual", color="black", linewidth=2)
    plt.plot(lstm_preds, label="LSTM Predicted", linestyle="--")
    plt.plot(attn_preds, label="Attention LSTM Predicted", linestyle="--")
    plt.title(f"{symbol} — LSTM & Attention LSTM vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    filename = "combined_lstm_attention_predictions.png"
    plt.savefig(filename)
    wandb.log({"Combined_LSTM_vs_Attention_vs_Actuals": wandb.Image(filename)})
    print(f"✅ Combined prediction plot saved: {filename}")

    # === Summary ===
    print("\n=============================")
    print(f"📊 All RMSEs for {symbol}:")
    for name, rmse in baseline_rmse.items():
        print(f"{name}: {rmse:.4f}")
    print(f"LSTM: {lstm_rmse:.4f}")
    print(f"Attention LSTM: {attn_rmse:.4f}")
    print("=============================\n")

    wandb.finish()


if __name__ == "__main__":
    compare_all_models("AAPL")

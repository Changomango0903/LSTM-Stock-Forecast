import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

from sklearn.preprocessing import StandardScaler

from data_loader import fetch_stock_data
from feature_engineering import add_technical_indicators, train_test_split
from lstm_models import LSTMModel
from train import scale_data

MODEL_PATH = 'models/your_trained_lstm.pt'

CONFIG = {
    'sequence_length': 60,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'project_name': 'stock_forecasting_pipeline'
}

SYMBOL = "AAPL"

def run_sweep():
    wandb.init(project=CONFIG['project_name'], name="threshold_sweep")

    df = fetch_stock_data(SYMBOL)
    df = add_technical_indicators(df)
    X_train, X_test, y_train, y_test = train_test_split(df, sequence_length=CONFIG['sequence_length'])
    X_train, X_test, y_train, y_test, X_scaler, y_scaler = scale_data(X_train, X_test, y_train, y_test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(
        input_size=X_train.shape[2],
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_test_tensor).squeeze().cpu().numpy()

    preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    prev_actuals = np.roll(y_actual, 1)
    prev_actuals[0] = y_actual[0]

    predicted_return = (preds - prev_actuals) / prev_actuals
    actual_return = (y_actual - prev_actuals) / prev_actuals

    thresholds = np.arange(0.0, 0.01, 0.001)
    results = []

    for THRESHOLD in thresholds:
        long_signals = np.where(predicted_return > THRESHOLD, 1, 0)
        long_strategy_return = actual_return * long_signals
        long_win_rate = np.mean(long_strategy_return > 0)
        long_sharpe = (long_strategy_return.mean() / long_strategy_return.std()) * np.sqrt(252) if long_strategy_return.std() > 0 else 0
        long_equity = np.prod(1 + long_strategy_return)

        ls_signals = np.where(predicted_return > THRESHOLD, 1, np.where(predicted_return < -THRESHOLD, -1, 0))
        ls_strategy_return = actual_return * ls_signals
        ls_win_rate = np.mean(ls_strategy_return > 0)
        ls_sharpe = (ls_strategy_return.mean() / ls_strategy_return.std()) * np.sqrt(252) if ls_strategy_return.std() > 0 else 0
        ls_equity = np.prod(1 + ls_strategy_return)

        results.append({
            'Threshold': THRESHOLD,
            'Long_Win_Rate': long_win_rate,
            'Long_Sharpe': long_sharpe,
            'Long_Equity': long_equity,
            'LS_Win_Rate': ls_win_rate,
            'LS_Sharpe': ls_sharpe,
            'LS_Equity': ls_equity
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("threshold_sweep_results.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Threshold'], results_df['Long_Sharpe'], label="Long Sharpe")
    plt.plot(results_df['Threshold'], results_df['LS_Sharpe'], label="Long-Short Sharpe")
    plt.xlabel("Threshold")
    plt.ylabel("Sharpe Ratio")
    plt.title(f"Sharpe Ratio vs Threshold — {SYMBOL}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sweep_sharpe.png")

    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Threshold'], results_df['Long_Equity'], label="Long Equity")
    plt.plot(results_df['Threshold'], results_df['LS_Equity'], label="Long-Short Equity")
    plt.xlabel("Threshold")
    plt.ylabel("Final Cumulative Return")
    plt.title(f"Equity vs Threshold — {SYMBOL}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sweep_equity.png")

    # Log best Sharpe
    best_long = results_df.loc[results_df['Long_Sharpe'].idxmax()]
    best_ls = results_df.loc[results_df['LS_Sharpe'].idxmax()]

    wandb.log({
        "Best_Long_Sharpe": best_long['Long_Sharpe'],
        "Best_Long_Threshold": best_long['Threshold'],
        "Best_LS_Sharpe": best_ls['LS_Sharpe'],
        "Best_LS_Threshold": best_ls['Threshold']
    })

    wandb.log({
        "Sweep_Sharpe_Plot": wandb.Image("sweep_sharpe.png"),
        "Sweep_Equity_Plot": wandb.Image("sweep_equity.png")
    })
    wandb.save("threshold_sweep_results.csv")
    wandb.finish()

    print("✅ threshold_sweep_results.csv, sweep_sharpe.png, sweep_equity.png saved and logged.")

if __name__ == "__main__":
    run_sweep()

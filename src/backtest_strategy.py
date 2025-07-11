import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import wandb

from sklearn.preprocessing import StandardScaler

from data_loader import fetch_stock_data
from feature_engineering import add_technical_indicators, train_test_split
from lstm_models import AttentionLSTM, LSTMModel
from train import scale_data
from evaluate import compute_finance_metrics

# === CONFIG ===
THRESHOLD = 0.001  # 0.1%
MODEL_PATH = 'models/trim-star-20_LSTM.pt'
CONFIG = {
    'sequence_length': 60,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'project_name': 'model_backtest',
    'start_date': '2020-07-11',   # ✅ New!
    'end_date': '2025-07-11'      # ✅ New!
    
}
SYMBOL = "AAPL"

def main():
    wandb.init(project=CONFIG['project_name'], config=CONFIG)

    df = fetch_stock_data(SYMBOL)
    df = df.loc[CONFIG['start_date']:CONFIG['end_date']]
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

    # === Position tracking ===
    position = []
    pos = 0  # Start Flat

    for pred in predicted_return:
        if pred > THRESHOLD:
            pos = 1  # Long
        elif pred < -THRESHOLD:
            pos = -1  # Short
        else:
            pos = 0  # Flat
        position.append(pos)

    position = np.array(position)
    strategy_return = actual_return * position
    equity_curve = np.cumprod(1 + strategy_return)

    # === Entries & Exits ===
    pos_change = np.diff(position, prepend=0)
    long_entries = (pos_change == 1)
    long_exits = (pos_change == -1) & (position == 0)
    short_entries = (pos_change == -1) & (position == -1)
    short_exits = (pos_change == 1) & (position == 0)

    # === Metrics ===
    win_rate_long = np.mean((actual_return > 0)[position == 1]) if np.any(position == 1) else np.nan
    win_rate_short = np.mean((actual_return < 0)[position == -1]) if np.any(position == -1) else np.nan
    sharpe = (strategy_return.mean() / strategy_return.std()) * np.sqrt(252) if strategy_return.std() > 0 else 0
    num_long_trades = np.sum(long_entries)
    num_short_trades = np.sum(short_entries)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()

    print(f"Win Rate (Long): {win_rate_long:.4f}")
    print(f"Win Rate (Short): {win_rate_short:.4f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Long Trades: {num_long_trades}, Short Trades: {num_short_trades}")
    print(f"Max Drawdown: {max_drawdown:.4f}")

    # === Trades DataFrame ===
    price_dates = df.index[-len(y_actual):]
    trades = pd.DataFrame({
        'Date': price_dates,
        'Predicted_Return': predicted_return,
        'Actual_Return': actual_return,
        'Position': position,
        'Strategy_Return': strategy_return,
        'Equity': equity_curve,
        'Long_Entry': long_entries,
        'Long_Exit': long_exits,
        'Short_Entry': short_entries,
        'Short_Exit': short_exits
    })
    trades.to_csv("backtest_longshort_trades.csv", index=False)

    # === Price Plot with entries/exits ===
    plt.figure(figsize=(14, 7))
    plt.plot(price_dates, y_actual, label="Close Price", linewidth=1)

    plt.scatter(price_dates[long_entries], y_actual[long_entries], color='green', marker='^', label="Long Entry", zorder=5)
    plt.scatter(price_dates[long_exits], y_actual[long_exits], color='green', marker='v', label="Long Exit", zorder=5)
    plt.scatter(price_dates[short_entries], y_actual[short_entries], color='red', marker='v', label="Short Entry", zorder=5)
    plt.scatter(price_dates[short_exits], y_actual[short_exits], color='red', marker='^', label="Short Exit", zorder=5)

    plt.title(f"{SYMBOL} Price with Long/Short Entries & Exits")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("price_signals_longshort.png")

    # === Equity Curve ===
    plt.figure(figsize=(14, 7))
    plt.plot(price_dates, equity_curve, label="Equity Curve", color='blue')
    plt.title(f"Backtest Equity Curve — {SYMBOL} (Long/Short)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("equity_curve_longshort.png")

    # === Log to W&B ===
    wandb.log({
        "Win_Rate_Long": win_rate_long,
        "Win_Rate_Short": win_rate_short,
        "Sharpe": sharpe,
        "Num_Long_Trades": num_long_trades,
        "Num_Short_Trades": num_short_trades,
        "Max_Drawdown": max_drawdown
    })
    wandb.log({
        "Price_with_LongShort_Signals": wandb.Image("price_signals_longshort.png"),
        "Equity_Curve_LongShort": wandb.Image("equity_curve_longshort.png")
    })
    wandb.save("backtest_longshort_trades.csv")
    wandb.finish()

    print("✅ Upgraded backtest files saved and logged: CSV + plots.")

if __name__ == "__main__":
    main()

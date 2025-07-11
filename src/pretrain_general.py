import wandb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from time import sleep

from data_loader import fetch_stock_data
from feature_engineering import add_technical_indicators
from lstm_models import LSTMModel, AttentionLSTM
from train import train_model

import torch


CONFIG = {
    'symbols': ['AAPL', 'MSFT', 'NVDA'],  # Add more tickers for full pretrain
    'sequence_length': 60,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 150,
    'project_name': 'general_stock_pretrain'
}


def build_sequences(df: pd.DataFrame, seq_len: int):
    """
    Build sequences for LSTM training with 5-day log return target.
    """
    feature_cols = [
        'Returns', 'SMA_20', 'SMA_50', 'EMA_20',
        'BB_Middle', 'BB_Upper', 'BB_Lower',
        'RSI', 'MACD', 'MACD_Signal',
        'Momentum', 'OBV', 'Tenkan_sen', 'Kijun_sen',
        'Senkou_span_A', 'Senkou_span_B', 'Chikou_span',
        'Volatility'
    ]

    data = df[feature_cols].values
    sequences, targets = [], []

    for i in range(seq_len, len(data) - 5):
        seq = data[i - seq_len:i]
        target = df['LogClose'].iloc[i + 5] - df['LogClose'].iloc[i]  # 5-day log return
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)


def load_and_prepare(symbols, seq_len):
    """
    Load stocks, engineer features, scale per stock, merge sequences.
    """
    X_all, y_all = [], []

    for symbol in tqdm(symbols, desc="Processing symbols"):
        print(f"Fetching data for {symbol}")
        print("Sleeping for 12 Seconds...")
        sleep(12)
        df = fetch_stock_data(symbol)
        df = add_technical_indicators(df)
        df['LogClose'] = np.log(df['close'])
        df['Returns'] = df['close'].pct_change()
        df.dropna(inplace=True)

        X_seq, y_seq = build_sequences(df, seq_len)

        # Scale features per stock
        samples, timesteps, features = X_seq.shape
        scaler = StandardScaler()
        X_flat = X_seq.reshape(-1, features)
        X_scaled = scaler.fit_transform(X_flat).reshape(samples, timesteps, features)

        X_all.append(X_scaled)
        y_all.append(y_seq)  # 5-day log returns: no scaling needed

    X_merged = np.concatenate(X_all, axis=0)
    y_merged = np.concatenate(y_all, axis=0)

    indices = np.arange(len(X_merged))
    np.random.shuffle(indices)
    X_merged = X_merged[indices]
    y_merged = y_merged[indices]

    split = int(len(X_merged) * 0.8)
    X_train, X_test = X_merged[:split], X_merged[split:]
    y_train, y_test = y_merged[:split], y_merged[split:]

    return X_train, X_test, y_train, y_test


def main():
    wandb.init(project=CONFIG['project_name'], config=CONFIG)

    X_train, X_test, y_train, y_test = load_and_prepare(CONFIG['symbols'], CONFIG['sequence_length'])

    input_size = X_train.shape[2]

    # === General LSTM ===
    lstm = LSTMModel(input_size, CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout'])
    print("\n=== Training General LSTM ===")
    train_model(
        lstm,
        X_train, y_train, X_test, y_test,
        CONFIG | {'model_name': 'General_LSTM'}
    )

    # === General Attention LSTM ===
    attn_lstm = AttentionLSTM(input_size, CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout'])
    print("\n=== Training General Attention LSTM ===")
    train_model(
        attn_lstm,
        X_train, y_train, X_test, y_test,
        CONFIG | {'model_name': 'General_Attention_LSTM'}
    )

    wandb.finish()


if __name__ == "__main__":
    main()
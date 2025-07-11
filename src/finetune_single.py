import wandb
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_loader import fetch_stock_data
from feature_engineering import add_technical_indicators
from lstm_models import LSTMModel
from train import train_model

CONFIG = {
    'sequence_length': 60,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.0005,
    'batch_size': 32,
    'epochs': 10,
    'project_name': 'finetune_stock_model',
    'model_name': 'AAPL_Finetune',
    'pretrained_model': 'models/bright-dream-3_General_LSTM.pt'
}


def build_sequences(df: np.ndarray, seq_len: int):
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

    for i in range(seq_len, len(data) - 1):
        seq = data[i - seq_len:i]
        target = df['close'].iloc[i + 1]  # Next-day raw price
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)


def main():
    wandb.init(project=CONFIG['project_name'], config=CONFIG)

    symbol = 'AAPL'
    df = fetch_stock_data(symbol)
    df = add_technical_indicators(df)
    df['Returns'] = df['close'].pct_change()
    df.dropna(inplace=True)

    X_seq, y_seq = build_sequences(df, CONFIG['sequence_length'])

    # === Scale features
    samples, timesteps, features = X_seq.shape
    X_scaler = StandardScaler()
    X_flat = X_seq.reshape(-1, features)
    X_scaled = X_scaler.fit_transform(X_flat).reshape(samples, timesteps, features)

    # === Scale target: next-day price
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_seq.reshape(-1, 1)).flatten()

    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    input_size = X_train.shape[2]

    lstm = LSTMModel(input_size, CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout'])
    lstm.load_state_dict(torch.load(CONFIG['pretrained_model']))

    print(f"\n=== Fine-tuning on {symbol} for next-day price ===")
    train_model(
        lstm,
        X_train, y_train, X_test, y_test,
        CONFIG,
        y_scaler=y_scaler
    )

    wandb.finish()


if __name__ == "__main__":
    main()

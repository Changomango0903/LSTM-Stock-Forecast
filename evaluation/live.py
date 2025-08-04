"""
Daily live prediction pipeline—strictly causal:
1) Fetch history up to yesterday
2) Denoise that history only
3) Append today’s open + carry-forward close
4) Compute lagged indicators & drop NaNs
5) Build final 60-day sequence
6) Log only the real input window (dates ≤ yesterday)
7) Load model/scalers, scale & predict
8) Record today’s prediction to CSV & W&B
9) Compute practical directional accuracy on past dates only
"""

import logging
import os
import numpy as np
import pandas as pd
import torch
import joblib
import wandb
from datetime import datetime, timedelta
from config import CONFIG
from data.alphavantage import fetch_historical, fetch_latest_open
from denoising.denoise import apply_denoising
from features.indicators import add_technical_indicators
from data.split import create_sequences
from models.lstm import LSTMModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_live_predict(symbol: str):
    logger.info("=== Live Predict Start ===")
    wandb.init(
        project=CONFIG["wandb_project"],
        name=f"live_{symbol}_{datetime.utcnow():%Y%m%dT%H%M}",
        reinit=True
    )

    # 1) Fetch history up to yesterday
    yesterday = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
    hist = fetch_historical(symbol, CONFIG["start_date"], yesterday)

    # 2) Denoise only on that past history
    hist["close"] = apply_denoising(hist["close"], CONFIG["denoising_method"])

    # 3) Append today’s open + carry-forward yesterday’s close, high, low, volume
    today = datetime.utcnow().date()
    open_price = fetch_latest_open(symbol)
    last = hist.iloc[-1]
    hist.loc[pd.Timestamp(today), ["open","high","low","close","volume"]] = [
        open_price, last["high"], last["low"], last["close"], last["volume"]
    ]

    # 4) Compute lagged indicators
    df = add_technical_indicators(hist)
    df = df[CONFIG["feature_cols"]]
    logger.debug(f"Post-features df shape: {df.shape}")

    # 5) Build sequence for prediction
    X, _, dates = create_sequences(df, CONFIG["sequence_length"])
    X_last = X[-1:,:,:]

    # 6) Log only the last 60 real days (exclude today’s placeholder)
    full_window = df.iloc[-CONFIG["sequence_length"]:]
    window_df = full_window.iloc[:-1]  # drop placeholder
    logger.info("Input window for model (last 60 real days):")
    for dt, row in window_df.iterrows():
        vals = " | ".join(f"{col}={row[col]:.4f}" for col in window_df.columns)
        logger.info(f"  {dt.date()} | {vals}")

    # 7) Load model & scalers
    model = LSTMModel(
        input_size=X_last.shape[2],
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"]
    )
    model_path = os.path.join(CONFIG["model_dir"], f"{symbol}_latest.pt")
    try:
        model.load_state_dict(torch.load(model_path))
    except RuntimeError:
        logger.error(
            "Model-input-size mismatch when loading checkpoint. "
            "Did you change CONFIG['feature_cols']? Please retrain."
        )
        raise
    model.eval()

    x_scaler = joblib.load(os.path.join(CONFIG["scaler_dir"], f"x_scaler_{symbol}.pkl"))
    y_scaler = joblib.load(os.path.join(CONFIG["scaler_dir"], f"y_scaler_{symbol}.pkl"))

    # 8) Scale & predict
    bs, sl, nf = X_last.shape
    Xf = X_last.reshape(-1, nf)
    Xs = x_scaler.transform(Xf).reshape(bs, sl, nf)
    with torch.no_grad():
        norm_pred = model(torch.tensor(Xs, dtype=torch.float32)).item()
    pred = y_scaler.inverse_transform([[norm_pred]])[0,0]
    logger.info(f"Predicted close for {symbol} on {today}: {pred:.2f}")

    # Record today’s prediction
    out = {"date": pd.Timestamp(today), "pred": pred}
    os.makedirs(os.path.dirname(CONFIG["live_csv"]), exist_ok=True)
    pd.DataFrame([out]).to_csv(
        CONFIG["live_csv"],
        mode="a",
        header=not os.path.exists(CONFIG["live_csv"]),
        index=False
    )
    wandb.log(out)

    # 9) Compute and log live directional accuracy based on past dates only
    live_df = pd.read_csv(CONFIG["live_csv"], parse_dates=["date"])
    past_df = live_df[live_df["date"].dt.date < datetime.utcnow().date()]
    if len(past_df) > 1:
        start_date = past_df["date"].min().strftime("%Y-%m-%d")
        end_date   = past_df["date"].max().strftime("%Y-%m-%d")
        hist2 = fetch_historical(symbol, start_date, end_date)
        actuals = [hist2.loc[pd.Timestamp(d), "close"] for d in past_df["date"]]
        past_df["actual"] = actuals

        da = (np.sign(past_df["pred"].diff().dropna())
              == np.sign(past_df["actual"].diff().dropna())).mean()
        wandb.log({"Live_DA": da})
        logger.info(f"Live directional accuracy: {da:.3f}")

    wandb.finish()
    logger.info("=== Live Predict End ===")

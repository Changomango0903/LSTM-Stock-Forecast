"""
Daily live prediction pipeline—strictly causal:
1) Fetch history up to yesterday
2) Denoise that history only
3) Append today’s open + carry-forward close
4) Compute lagged indicators & drop NaNs
5) Build final 60-day sequence
6) Log the raw input window (dates + features)
7) Load model/scalers, scale & predict
8) Record to CSV & W&B
9) Compute practical directional accuracy
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

    # 3) Append today’s open + carry-forward
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

    # 5) Build sequence
    X, _, dates = create_sequences(df, CONFIG["sequence_length"])
    X_last = X[-1:,:,:]

    # 6) Log the raw input window
    #    Show the last `sequence_length` rows of df (features + timestamp)
    window_df = df.iloc[-CONFIG["sequence_length"]:]
    logger.info("Input window for model (last 60 days):")
    for dt, row in window_df.iterrows():
        logger.info(f"  {dt.date()} | " + " | ".join(f"{col}={row[col]:.4f}" for col in window_df.columns))

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
    except RuntimeError as e:
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
    logger.info(f"Predicted next close for {symbol} on {(pd.to_datetime(dates[-1]) + pd.Timedelta(days=1)).date()}: {pred:.2f}")

    # 9) Record to CSV & W&B
    # last_date = pd.to_datetime(dates[-1])
    # next_date = last_date + pd.Timedelta(days=1)
    
    # We want to label this prediction as “today”
    # since we’re running after today’s open and forecasting for today
    next_date = pd.Timestamp(today)
    
    out = {"date": next_date, "pred": pred}

    os.makedirs(os.path.dirname(CONFIG["live_csv"]), exist_ok=True)
    pd.DataFrame([out]).to_csv(
        CONFIG["live_csv"],
        mode="a",
        header=not os.path.exists(CONFIG["live_csv"]),
        index=False
    )
    wandb.log(out)

    # 10) Compute and log live directional accuracy
    live_df = pd.read_csv(CONFIG["live_csv"], parse_dates=["date"])
    if len(live_df) > 1:
        hist2 = fetch_historical(
            symbol,
            live_df["date"].min().strftime("%Y-%m-%d"),
            live_df["date"].max().strftime("%Y-%m-%d")
        )
        actuals = [hist2.loc[pd.Timestamp(d), "close"] for d in live_df["date"]]
        live_df["actual"] = actuals
        da = (np.sign(live_df["pred"].diff().dropna())
              == np.sign(live_df["actual"].diff().dropna())).mean()
        wandb.log({"Live_DA": da})
        logger.info(f"Live directional accuracy: {da:.3f}")

    wandb.finish()
    logger.info("=== Live Predict End ===")

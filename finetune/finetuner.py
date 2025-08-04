"""
Incremental fine-tuning on the most recent holdout window:
1) Fetch last 2×holdout_size days
2) Denoise & lagged indicators
3) Build sequences
4) Split: first for context, last holdout_size for new data
5) Scale new data with existing scalers
6) Train 3 epochs at low LR, save updated model & y-scaler
"""

import logging, os, torch, joblib, numpy as np
from datetime import datetime, timedelta
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from config import CONFIG
from data.alphavantage import fetch_historical
from denoising.denoise import apply_denoising
from features.indicators import add_technical_indicators
from data.split import create_sequences
from data.scaler import scale_data
from models.lstm import LSTMModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_finetune(symbol: str, epochs:int=3, lr:float=1e-4):
    logger.info("=== Fine-tune Start ===")
    end_dt = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
    start_dt = (datetime.utcnow().date() - timedelta(days=CONFIG["holdout_size"]*2)).isoformat()
    hist = fetch_historical(symbol, start_dt, end_dt)

    hist["close"] = apply_denoising(hist["close"], CONFIG["denoising_method"])
    df = add_technical_indicators(hist)
    df = df[CONFIG["feature_cols"]]
    X, y, _ = create_sequences(df, CONFIG["sequence_length"])

    split = len(X) - CONFIG["holdout_size"]
    X_ctx, X_new = X[:split], X[split:]
    y_ctx, y_new = y[:split], y[split:]
    X_ctx_s, X_new_s, y_ctx_s, y_new_s, _, y_sc = scale_data(X_ctx, X_new, y_ctx, y_new)

    model = LSTMModel(X_new_s.shape[2], CONFIG["hidden_size"],
                      CONFIG["num_layers"], CONFIG["dropout"])
    model.load_state_dict(torch.load(os.path.join(CONFIG["model_dir"], f"{symbol}_latest.pt")))
    model.train()

    ds = TensorDataset(torch.tensor(X_new_s,dtype=torch.float32),
                       torch.tensor(y_new_s,dtype=torch.float32))
    dl = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = MSELoss()

    for ep in range(1, epochs+1):
        losses = []
        for xb, yb in dl:
            pred = model(xb).squeeze()
            loss = crit(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        logger.info(f"Ep {ep}/{epochs}  AvgLoss={np.mean(losses):.5f}")

    torch.save(model.state_dict(), os.path.join(CONFIG["model_dir"], f"{symbol}_latest.pt"))
    joblib.dump(y_sc, os.path.join(CONFIG["scaler_dir"], f"y_scaler_{symbol}.pkl"))
    logger.info("=== Fine-tune End ===")

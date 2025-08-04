"""
Train LSTMModel with W&B logging and holdout validation.
Saves best model by lowest holdout RMSE. Optionally returns history.
"""

import os
import torch
import wandb
import logging
import numpy as np
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from config import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_model(model, X_train, y_train, X_test, y_test, y_scaler, test_dates,
                return_history: bool = False):
    """
    Trains for CONFIG['epochs'], logs train_loss, holdout RMSE & DA.
    If return_history=True, returns (best_rmse, preds, actuals, dates, history),
    where history is a dict with lists: train_losses, holdout_rmses, holdout_das.
    Otherwise returns (best_rmse, preds, actuals, dates).
    """
    from models.lstm import LSTMModel
    input_size = X_train.shape[2]
    model = model or LSTMModel(input_size, CONFIG["hidden_size"],
                               CONFIG["num_layers"], CONFIG["dropout"])
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    crit = MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))
    train_dl = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False)

    best_rmse = float("inf")
    wandb.watch(model, log="all", log_freq=10)

    # prepare history lists if needed
    if return_history:
        history = {"train_losses": [], "holdout_rmses": [], "holdout_das": []}

    for ep in range(1, CONFIG["epochs"]+1):
        # Training
        model.train()
        losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        train_loss = np.mean(losses)

        # Validation
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                p = model(xb).cpu().numpy()
                all_p.append(p); all_t.append(yb.cpu().numpy())
        preds_norm = np.concatenate(all_p)
        trues_norm = np.concatenate(all_t)
        actuals = y_scaler.inverse_transform(trues_norm.reshape(-1,1)).flatten()
        preds   = y_scaler.inverse_transform(preds_norm.reshape(-1,1)).flatten()

        rmse = np.sqrt(np.mean((preds-actuals)**2))
        da   = np.mean(np.sign(preds[1:]-preds[:-1]) == np.sign(actuals[1:]-actuals[:-1]))

        logger.info(f"Epoch {ep}/{CONFIG['epochs']}  TrainLoss={train_loss:.4f}  RMSE={rmse:.4f}  DA={da:.4f}")
        wandb.log({"Train_Loss": train_loss, "Holdout_RMSE": rmse, "Holdout_DA": da, "epoch": ep})

        if return_history:
            history["train_losses"].append(train_loss)
            history["holdout_rmses"].append(rmse)
            history["holdout_das"].append(da)

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(),
                       os.path.join(CONFIG["model_dir"], f"{CONFIG['default_symbol']}_latest.pt"))

    if return_history:
        return best_rmse, preds, actuals, test_dates, history
    return best_rmse, preds, actuals, test_dates

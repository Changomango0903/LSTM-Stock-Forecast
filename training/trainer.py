"""
training/trainer.py

Purpose:
--------
Contains core training loop for LSTM and Attention LSTM models with wandb logging,
real-time progress bars, plotting, and checkpoint saving.

Inputs:
-------
- model (nn.Module): PyTorch LSTM or AttentionLSTM
- X_train, y_train, X_test, y_test: ndarray
- config (dict): model + logging hyperparameters
- y_scaler (StandardScaler): optional inverse-scaling for output

Outputs:
--------
- rmse (float)
- preds_np (np.ndarray): inverse-scaled predictions
- y_test_np (np.ndarray): inverse-scaled targets
- history (dict): train and val losses per epoch
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import os

from utils.plots import plot_predictions, plot_loss_curve


def train_model(model, X_train, y_train, X_test, y_test, config, y_scaler=None, test_dates=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)

        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(train_loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
            preds = model(X_test_tensor).squeeze()
            val_loss = criterion(preds, y_test_tensor).item()

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)

        model_prefix = config['model_name']  # e.g., 'LSTM' or 'Attention_LSTM'

        wandb.log({
            f"{model_prefix}_Train_Loss": avg_train_loss,
            f"{model_prefix}_Val_Loss": val_loss,
            f"{model_prefix}_Epoch": epoch + 1
        })

        print(f"[Epoch {epoch+1}/{config['epochs']}] Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f}")

    # === Final RMSE ===
    rmse = torch.sqrt(criterion(preds, y_test_tensor)).item()
    wandb.log({"Test RMSE": rmse})

    # === Inverse scale ===
    preds_np = preds.detach().cpu().numpy()
    y_test_np = y_test_tensor.detach().cpu().numpy()
    if y_scaler:
        preds_np = y_scaler.inverse_transform(preds_np.reshape(-1, 1)).flatten()
        y_test_np = y_scaler.inverse_transform(y_test_np.reshape(-1, 1)).flatten()

    # === Plots ===
    plot_predictions(preds_np, y_test_np, config['model_name'], test_dates)
    plot_loss_curve(history, config['model_name'])

    # === Save model ===
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{wandb.run.name}_{config['model_name']}.pt"
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)

    return rmse, preds_np, y_test_np, history

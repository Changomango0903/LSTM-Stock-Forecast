import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler

def scale_data(X_train, X_test, y_train, y_test):
    """
    Standardize X and y for stable LSTM training.
    """
    samples, timesteps, features = X_train.shape

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Flatten to apply scaler
    X_train_flat = X_train.reshape(-1, features)
    X_test_flat = X_test.reshape(-1, features)

    X_train_scaled = X_scaler.fit_transform(X_train_flat).reshape(samples, timesteps, features)
    X_test_scaled = X_scaler.transform(X_test_flat).reshape(X_test.shape[0], timesteps, features)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler


def train_model(model, X_train, y_train, X_test, y_test, config, y_scaler=None):
    """
    Train LSTM or Attention LSTM with wandb logging, terminal progress, and final predictions.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)

        for X_batch, y_batch in batch_progress:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_progress.set_postfix({'Batch Loss': loss.item()})

        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
            preds = model(X_test_tensor).squeeze()
            val_loss = criterion(preds, y_test_tensor).item()

        wandb.log({
            "Train Loss": avg_train_loss,
            "Val Loss": val_loss,
            "epoch": epoch + 1
        })

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # === Compute final RMSE ===
    rmse = torch.sqrt(criterion(preds, y_test_tensor)).item()
    wandb.log({"Test RMSE": rmse})

    # === Inverse scale if scaler is provided ===
    preds_np = preds.detach().cpu().numpy()
    y_test_np = y_test_tensor.detach().cpu().numpy()

    if y_scaler is not None:
        preds_np = y_scaler.inverse_transform(preds_np.reshape(-1, 1)).flatten()
        y_test_np = y_scaler.inverse_transform(y_test_np.reshape(-1, 1)).flatten()

    # === Loss curve plot ===
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label="Train Loss")
    plt.plot(history['val_loss'], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{config['model_name']} Loss Curve")
    plt.legend()
    plt.tight_layout()
    loss_curve_file = f"{config['model_name']}_loss_curve.png"
    plt.savefig(loss_curve_file)
    wandb.log({f"{config['model_name']}_Loss_Curve": wandb.Image(loss_curve_file)})

    # === Predictions vs Actual plot ===
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_np, label="Actual")
    plt.plot(preds_np, label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"{config['model_name']} Predictions vs Actual")
    plt.legend()
    plt.tight_layout()
    pred_plot_file = f"{config['model_name']}_predictions.png"
    plt.savefig(pred_plot_file)
    wandb.log({f"{config['model_name']}_Predictions_vs_Actuals": wandb.Image(pred_plot_file)})

    # === Save model artifact ===
    model_file = f"models/{wandb.run.name}_{config['model_name']}.pt"
    torch.save(model.state_dict(), model_file)
    wandb.save(model_file)

    return rmse, preds_np, y_test_np, history
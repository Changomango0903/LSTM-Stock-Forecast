"""
Plot predictions vs. actuals.
"""

import matplotlib.pyplot as plt

def plot_pred_vs_actual(dates, preds, actuals, title="Pred vs Actual"):
    plt.figure()
    plt.plot(dates, actuals, label="Actual")
    plt.plot(dates, preds,   label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pred_vs_actual(dates, preds, actuals, title="Pred vs Actual"):
    plt.figure()
    plt.plot(dates, actuals, label="Actual")
    plt.plot(dates, preds,   label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_training_curves(epochs, train_losses, holdout_rmses, holdout_das):
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, holdout_rmses, label="Holdout RMSE")
    plt.title("Training & Holdout Metrics")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(epochs, holdout_das, label="Holdout DA")
    plt.title("Holdout Directional Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("DA")
    plt.legend()
    plt.tight_layout()
    plt.show()
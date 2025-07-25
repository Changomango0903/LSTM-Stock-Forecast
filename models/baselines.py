"""
models/baselines.py

Purpose:
--------
Train classic ML models (Linear, Ridge, SVR, etc.) as benchmarks.

Inputs:
-------
- X_train, y_train, X_test, y_test (np.ndarray)

Outputs:
--------
- results (dict): model_name → RMSE
- predictions (dict): model_name → (preds, y_test)
"""

from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os


def train_baseline_models(X_train, y_train, X_test, y_test, test_dates):
    os.makedirs("plots", exist_ok=True)
    X_train_last = X_train[:, -1, :]
    X_test_last = X_test[:, -1, :]

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "SVR": SVR(kernel='rbf'),
        "DecisionTree": DecisionTreeRegressor(max_depth=5),
        "RandomForest": RandomForestRegressor(n_estimators=100),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100)
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train_last, y_train)
        preds = model.predict(X_test_last)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = rmse
        predictions[name] = (preds, y_test)
        plt.figure(figsize=(10, 5))
        plt.plot(test_dates, y_test, label="Actual")
        plt.plot(test_dates, preds, label=f"{name} Predicted")
        plt.title(f"{name} Predictions vs Actual")
        plt.legend()
        plt.tight_layout()
        filename = os.path.join("plots", f"{name}_predictions.png")
        plt.savefig(filename)
        wandb.log({f"{name}_Predictions_vs_Actuals": wandb.Image(filename)})

    return results, predictions

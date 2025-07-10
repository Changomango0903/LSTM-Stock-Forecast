from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import wandb

def train_baseline_models(X_train, y_train, X_test, y_test):
    """
    Train multiple classic ML regressors on flattened LSTM data (last timestep only).
    Args:
        X_train (np.array): Shape [samples, timesteps, features]
        y_train (np.array)
        X_test (np.array)
        y_test (np.array)
    Returns:
        dict: Model name -> RMSE
    """

    # For classic regressors, use only last timestep as snapshot features
    X_train_flat = X_train[:, -1, :]
    X_test_flat = X_test[:, -1, :]

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        "DecisionTree": DecisionTreeRegressor(max_depth=5),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_flat, y_train)
        preds = model.predict(X_test_flat)
        rmse = root_mean_squared_error(y_test, preds)
        results[name] = rmse

    predictions = {}

    for name, model in models.items():
        model.fit(X_train_flat, y_train)
        preds = model.predict(X_test_flat)
        rmse = root_mean_squared_error(y_test, preds)
        results[name] = rmse

        predictions[name] = (preds, y_test)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(preds, label=f"{name} Predicted")
        plt.plot(y_test, label="Actual")
        plt.legend()
        plt.title(f"{name} Predictions vs Actuals")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.tight_layout()
        filename = f"{name}_preds.png"
        plt.savefig(filename)
        wandb.log({f"{name}_Predictions_vs_Actuals": wandb.Image(filename)})

    return results, predictions

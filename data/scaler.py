"""
data/scaler.py

Purpose:
--------
Scales input features and target outputs for stable LSTM training.

Inputs:
-------
- X_train, X_test: ndarray [samples, timesteps, features]
- y_train, y_test: ndarray [samples]

Outputs:
--------
- Scaled X_train, X_test, y_train, y_test
- Fitted scalers for inverse transformation
"""

from sklearn.preprocessing import StandardScaler
import numpy as np


def scale_data(X_train, X_test, y_train, y_test):
    samples, timesteps, features = X_train.shape

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_flat = X_train.reshape(-1, features)
    X_test_flat = X_test.reshape(-1, features)

    X_train_scaled = X_scaler.fit_transform(X_train_flat).reshape(samples, timesteps, features)
    X_test_scaled = X_scaler.transform(X_test_flat).reshape(X_test.shape[0], timesteps, features)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler

"""
Scale train/test splits using StandardScaler.
Persists scalers and validates no NaNs.
"""

import os
import joblib
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def scale_data(X_train, X_test, y_train, y_test):
    """
    Fits feature & target scalers on training data, transforms both splits,
    saves scalers under CONFIG["scaler_dir"], and returns scaled arrays + scalers.
    """
    ns, sl, nf = X_train.shape
    logger.info(f"Scaling data: {ns} train samples, {X_test.shape[0]} test samples")
    Xtr = X_train.reshape(-1, nf)
    Xte = X_test.reshape(-1, nf)

    x_scaler = StandardScaler().fit(Xtr)
    X_train_s = x_scaler.transform(Xtr).reshape(ns, sl, nf)
    X_test_s  = x_scaler.transform(Xte).reshape(*X_test.shape)

    y_scaler = StandardScaler().fit(y_train.reshape(-1,1))
    y_train_s = y_scaler.transform(y_train.reshape(-1,1)).flatten()
    y_test_s  = y_scaler.transform(y_test.reshape(-1,1)).flatten()

    os.makedirs(CONFIG["scaler_dir"], exist_ok=True)
    joblib.dump(x_scaler, os.path.join(CONFIG["scaler_dir"], f"x_scaler_{CONFIG['default_symbol']}.pkl"))
    joblib.dump(y_scaler, os.path.join(CONFIG["scaler_dir"], f"y_scaler_{CONFIG['default_symbol']}.pkl"))

    for arr,name in [(X_train_s,"X_train_s"),(X_test_s,"X_test_s"),(y_train_s,"y_train_s"),(y_test_s,"y_test_s")]:
        if np.isnan(arr).any():
            raise ValueError(f"NaNs detected in {name}")

    return X_train_s, X_test_s, y_train_s, y_test_s, x_scaler, y_scaler

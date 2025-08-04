"""
Central configuration for the pipeline:
- Data parameters
- Model hyperparameters
- AlphaVantage API settings
- Denoising choice
- Artifact paths
- W&B project name
"""

import os

CONFIG = {
    # Data & sequences
    "sequence_length": 60,
    "holdout_size": 250,
    "start_date": "2020-01-01",
    "end_date": "2025-08-01",
    "default_symbol": "AAPL",

    # Model hyperparameters
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 130,

    # AlphaVantage
    "av_api_key": os.getenv("ALPHAVANTAGE_API_KEY", ""),
    "av_base_url": "https://www.alphavantage.co/query",

    # Denoising method: "raw", "ema", or "kalman"
    "denoising_method": "kalman",
    
    #Choose features
    "feature_cols": [
        "close",
        "SMA_20",
        "SMA_50",
        "RSI_14",
        "Volatility"
    ],
    
    # Artifacts
    "model_dir": "artifacts/models",
    "scaler_dir": "artifacts/scalers",
    "live_csv": "artifacts/live_results.csv",

    # Weights & Biases
    "wandb_project": "stock_forecasting_live",
}

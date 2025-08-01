"""
Purpose:
--------
Centralized configuration used across training, evaluation, and model components.

Contains model hyperparameters, sequence settings, API keys (via environment), 
project naming for wandb, etc.

Usage:
------
Import as:
    from config import CONFIG

Returns:
--------
CONFIG (dict): Central configuration dictionary for the pipeline.
"""

CONFIG = {
    # Model Training
    'sequence_length': 60,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 150,

    # WandB
    'project_name': 'stock_forecasting_holdout_evaluation',

    # Data Range
    'start_date': '2019-01-01',
    'end_date': '2025-01-01',

    # Default Symbol
    'default_symbol': 'AAPL',
}

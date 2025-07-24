"""
models/lstm.py

Purpose:
--------
Defines a basic LSTM model for time-series forecasting.

Inputs:
-------
- input_size (int): Number of input features per timestep.
- hidden_size (int): Number of hidden units in LSTM.
- num_layers (int): Number of stacked LSTM layers.
- dropout (float): Dropout rate.

Outputs:
--------
- forward(x): Forecasts a single value from a sequence.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Predict next-day value using LSTM output from final timestep.
        Args:
            x (Tensor): shape [batch_size, sequence_length, input_size]
        Returns:
            Tensor: shape [batch_size, 1]
        """
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]  # final timestep
        return self.fc(last_step)

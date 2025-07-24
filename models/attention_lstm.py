"""
models/attention_lstm.py

Purpose:
--------
Defines an LSTM model enhanced with attention mechanism.

Inputs:
-------
- input_size (int): Number of input features per timestep.
- hidden_size (int): LSTM hidden units.
- num_layers (int): Number of LSTM layers.
- dropout (float): Dropout rate.

Outputs:
--------
- forward(x): Attention-weighted forecast for next timestep.
"""

import torch
import torch.nn as nn


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.attn_layer = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Apply attention over all LSTM outputs before final prediction.
        Args:
            x (Tensor): shape [batch_size, sequence_length, input_size]
        Returns:
            Tensor: shape [batch_size, 1]
        """
        output, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn_layer(output), dim=1)
        context = torch.sum(attn_weights * output, dim=1)
        return self.fc(context)

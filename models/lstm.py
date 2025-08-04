"""
LSTM regressor for next-day close.
"""

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Args:
      input_size: number of features
      hidden_size, num_layers, dropout: model hyperparams
    Forward:
      expects (batch, seq_len, input_size) → returns (batch,) predictions
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.dropout(last)).squeeze(-1)

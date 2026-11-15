# -*- coding: utf-8 -*-
"""RNN 预测类模型。"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch.nn as nn

from models.common.forecast_runner import run_forecasting

MODEL_NAME = "RNN"
PREFIX = "rnn_swat"


class RNNForecaster(nn.Module):
    def __init__(self, n_channels, seq_len, pred_len=1,
                 hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.rnn = nn.RNN(
            input_size=n_channels, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True, nonlinearity="tanh")
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, pred_len * n_channels)

    def forward(self, x):
        h_seq, _ = self.rnn(x)
        h_last = self.norm(h_seq[:, -1, :])
        return self.head(h_last).view(-1, self.pred_len, self.n_channels)


def build_model(n_channels, seq_len, pred_len, data=None):
    return RNNForecaster(n_channels, seq_len, pred_len)


def main():
    run_forecasting(MODEL_NAME, PREFIX, build_model,
                    description="RNN 预测类模型（SWaT）")


if __name__ == "__main__":
    main()

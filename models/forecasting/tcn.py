# -*- coding: utf-8 -*-
"""TCN 预测类模型。"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from models.common.forecast_runner import run_forecasting

MODEL_NAME = "TCN"
PREFIX = "tcn_swat"


class TCNBlock(nn.Module):
    def __init__(self, ch, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(-1)]
        return self.drop(torch.relu(self.norm(out + x)))


class TCNForecaster(nn.Module):
    def __init__(self, n_channels, seq_len, pred_len=1, hidden=64,
                 dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.input_proj = nn.Conv1d(n_channels, hidden, 1)
        self.blocks = nn.Sequential(
            TCNBlock(hidden, dilation=1, dropout=dropout),
            TCNBlock(hidden, dilation=2, dropout=dropout),
            TCNBlock(hidden, dilation=4, dropout=dropout),
            TCNBlock(hidden, dilation=8, dropout=dropout),
        )
        self.head = nn.Linear(hidden, pred_len * n_channels)

    def forward(self, x):
        h = self.input_proj(x.transpose(1, 2))
        h = self.blocks(h)
        out = self.head(h[:, :, -1])
        return out.view(-1, self.pred_len, self.n_channels)


def build_model(n_channels, seq_len, pred_len, data=None):
    return TCNForecaster(n_channels, seq_len, pred_len)


def main():
    run_forecasting(MODEL_NAME, PREFIX, build_model,
                    description="TCN 预测类模型（SWaT）")


if __name__ == "__main__":
    main()

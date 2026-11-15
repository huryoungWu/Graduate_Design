# -*- coding: utf-8 -*-
"""ModernTCN 预测类模型（简化版）。"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from models.common.forecast_runner import run_forecasting

MODEL_NAME = "ModernTCN"
PREFIX = "moderntcn_swat"


class ModernTCNForecaster(nn.Module):
    """深度可分离大核卷积 + 逐点卷积。"""

    def __init__(self, n_channels, seq_len, pred_len=1, hidden=64,
                 kernel_size=5, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.dwconv = nn.Conv1d(n_channels, n_channels,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2,
                                groups=n_channels)
        self.pwconv = nn.Conv1d(n_channels, hidden, kernel_size=1)
        self.norm = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, pred_len * n_channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = torch.relu(self.dwconv(x))
        h = torch.relu(self.pwconv(h))
        h = self.drop(self.norm(h))[:, :, -1]
        out = self.head(h)
        return out.view(-1, self.pred_len, self.n_channels)


def build_model(n_channels, seq_len, pred_len, data=None):
    return ModernTCNForecaster(n_channels, seq_len, pred_len)


def main():
    run_forecasting(MODEL_NAME, PREFIX, build_model,
                    description="ModernTCN 预测类模型（SWaT）")


if __name__ == "__main__":
    main()

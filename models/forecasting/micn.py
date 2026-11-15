# -*- coding: utf-8 -*-
"""MICN 预测类模型（简化版）。"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from models.common.forecast_runner import run_forecasting

MODEL_NAME = "MICN"
PREFIX = "micn_swat"


class MICNForecaster(nn.Module):
    """局部卷积 + 全局卷积 + 融合，最后时刻接线性头。"""

    def __init__(self, n_channels, seq_len, pred_len=1, hidden=64,
                 dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.local_conv = nn.Conv1d(n_channels, hidden, kernel_size=3,
                                    padding=1)
        self.global_conv = nn.Conv1d(n_channels, hidden, kernel_size=7,
                                     padding=3)
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1),
            nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, pred_len * n_channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        h_local = torch.relu(self.local_conv(x))
        h_global = torch.relu(self.global_conv(x))
        h = self.fuse(torch.cat([h_local, h_global], dim=1))[:, :, -1]
        out = self.head(h)
        return out.view(-1, self.pred_len, self.n_channels)


def build_model(n_channels, seq_len, pred_len, data=None):
    return MICNForecaster(n_channels, seq_len, pred_len)


def main():
    run_forecasting(MODEL_NAME, PREFIX, build_model,
                    description="MICN 预测类模型（SWaT）")


if __name__ == "__main__":
    main()

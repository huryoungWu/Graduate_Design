# -*- coding: utf-8 -*-
"""TimesNet 预测类模型（简化版）。"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from models.common.forecast_runner import run_forecasting

MODEL_NAME = "TimesNet"
PREFIX = "timesnet_swat"


class TimesNetForecaster(nn.Module):
    """时域卷积分支 + FFT 频域分支，融合后线性头。"""

    def __init__(self, n_channels, seq_len, pred_len=1, hidden=64,
                 dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.time_conv = nn.Conv1d(n_channels, hidden, kernel_size=3,
                                   padding=1)
        self.freq_conv = nn.Conv1d(n_channels, hidden, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, pred_len * n_channels)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        h_time = torch.relu(self.time_conv(x_t))[:, :, -1]
        x_fft = torch.fft.rfft(x_t, dim=-1).abs()
        h_freq = torch.relu(self.freq_conv(x_fft))[:, :, -1]
        h = self.fuse(torch.cat([h_time, h_freq], dim=-1))
        out = self.head(h)
        return out.view(-1, self.pred_len, self.n_channels)


def build_model(n_channels, seq_len, pred_len, data=None):
    return TimesNetForecaster(n_channels, seq_len, pred_len)


def main():
    run_forecasting(MODEL_NAME, PREFIX, build_model,
                    description="TimesNet 预测类模型（SWaT）")


if __name__ == "__main__":
    main()

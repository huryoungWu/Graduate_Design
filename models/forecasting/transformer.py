# -*- coding: utf-8 -*-
"""Transformer 预测类模型。"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from models.common.forecast_runner import run_forecasting

MODEL_NAME = "Transformer"
PREFIX = "transformer_swat"


class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_channels, seq_len, pred_len=1,
                 d_model=128, nhead=8, num_layers=3, dim_ff=256,
                 dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len * n_channels)

    def forward(self, x):
        h = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])
        return self.head(h).view(-1, self.pred_len, self.n_channels)


def build_model(n_channels, seq_len, pred_len, data=None):
    return TimeSeriesTransformer(n_channels, seq_len, pred_len)


def main():
    run_forecasting(MODEL_NAME, PREFIX, build_model,
                    description="Transformer 预测类模型（SWaT）")


if __name__ == "__main__":
    main()

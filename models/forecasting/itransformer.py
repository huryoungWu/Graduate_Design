# -*- coding: utf-8 -*-
"""iTransformer 预测类模型（SWaT）。

只保留原生 iTransformer：变量维 token + 标准 Transformer 编码器。
异常检测统一走 Stage 1（保存预测） + Stage 2（残差/阈值/评估）。
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from models.common.forecast_runner import run_forecasting

MODEL_NAME = "iTransformer"
PREFIX = "itransformer_swat"


class ITransformer(nn.Module):
    """原生 iTransformer：变量维 token + 标准 Transformer 编码器。"""

    def __init__(self, n_channels, seq_len, pred_len=1,
                 d_model=128, nhead=8, num_layers=3, dim_ff=256,
                 dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.input_proj = nn.Linear(seq_len, d_model)
        self.var_emb = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x):
        h = self.input_proj(x.transpose(1, 2)) + self.var_emb[:, :x.size(2), :]
        h = self.encoder(h)
        h = self.norm(h)
        return self.head(h).transpose(1, 2)


def build_model(n_channels, seq_len, pred_len, data=None):
    return ITransformer(n_channels, seq_len, pred_len)


def main():
    run_forecasting(
        MODEL_NAME, PREFIX, build_model,
        description="iTransformer 预测类模型（SWaT）")


if __name__ == "__main__":
    main()

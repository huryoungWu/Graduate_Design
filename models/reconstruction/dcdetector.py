# -*- coding: utf-8 -*-
"""DCdetector：双编码器表示对比异常检测。"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.recon_runner import run_reconstruction

MODEL_NAME = "DCdetector"
PREFIX = "dcdetector_swat"

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 128
DROPOUT = 0.1
TEMPERATURE = 0.1


class EncoderBranch(nn.Module):
    def __init__(self, n_channels, win_size, d_model, nhead,
                 num_layers, dim_ff, dropout):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, win_size, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True,
            activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        return self.norm(self.encoder(h))


class DCdetector(nn.Module):
    """双分支编码器 + 表示对比；测试时用两路表示差异作为异常分数。"""

    def __init__(self, n_channels, win_size, d_model=64, nhead=4,
                 num_layers=2, dim_ff=128, dropout=0.1,
                 temperature=0.1):
        super().__init__()
        self.branch1 = EncoderBranch(
            n_channels, win_size, d_model, nhead, num_layers, dim_ff, dropout)
        self.branch2 = EncoderBranch(
            n_channels, win_size, d_model, nhead, num_layers, dim_ff, dropout)
        self.temperature = temperature

    def forward(self, x):
        return self.branch1(x), self.branch2(x)

    def training_loss(self, x):
        z1, z2 = self(x)
        return contrastive_loss(z1, z2, self.temperature)

    def score_batch(self, x):
        z1, z2 = self(x)
        diff = 1 - F.cosine_similarity(z1, z2, dim=-1)
        return {"window": {
            "score_max": diff.max(dim=1).values,
            "score_mean": diff.mean(dim=1),
        }}


def contrastive_loss(z1, z2, temperature=0.1):
    """逐时间点对比：同一时刻两路表示应相似。"""
    b, length, d = z1.shape
    z1 = F.normalize(z1, dim=-1).reshape(b * length, d)
    z2 = F.normalize(z2, dim=-1).reshape(b * length, d)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(b * length, device=z1.device)
    return F.cross_entropy(logits, labels)


def build_model(n_channels, seq_len, data=None):
    return DCdetector(n_channels, seq_len, D_MODEL, NHEAD, NUM_LAYERS,
                      DIM_FF, DROPOUT, TEMPERATURE)


def main():
    run_reconstruction(MODEL_NAME, PREFIX, build_model,
                       description="DCdetector（SWaT）",
                       batch_size=128, plot_level="window")


if __name__ == "__main__":
    main()

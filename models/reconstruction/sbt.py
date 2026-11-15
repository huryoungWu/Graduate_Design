# -*- coding: utf-8 -*-
"""SBT：自监督 Barlow Twins 表示对比异常检测。"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.recon_runner import run_reconstruction

MODEL_NAME = "SBT"
PREFIX = "sbt_swat"

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 128
DROPOUT = 0.1
LAMBDA_BT = 5e-3
NOISE_STD = 0.05
MASK_RATIO = 0.15


class Encoder(nn.Module):
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


class SBT(nn.Module):
    """共享编码器 + 两个增强视图，Barlow Twins 损失；测试用视图差异。"""

    def __init__(self, n_channels, win_size, d_model=64, nhead=4,
                 num_layers=2, dim_ff=128, dropout=0.1,
                 lambda_bt=5e-3, noise_std=0.05, mask_ratio=0.15):
        super().__init__()
        self.encoder = Encoder(
            n_channels, win_size, d_model, nhead, num_layers, dim_ff, dropout)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.lambda_bt = lambda_bt
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio

    def forward(self, x):
        return self.projector(self.encoder(x))

    def augment(self, x, mode="noise"):
        if mode == "noise":
            return x + torch.randn_like(x) * self.noise_std
        if mode == "mask":
            mask = (torch.rand_like(x) > self.mask_ratio).float()
            return x * mask
        return x

    def training_loss(self, x):
        v1 = self.augment(x, "noise")
        v2 = self.augment(x, "mask")
        return barlow_twins_loss(self(v1), self(v2), self.lambda_bt)

    def score_batch(self, x):
        v1 = self.augment(x, "noise")
        v2 = self.augment(x, "mask")
        diff = 1 - F.cosine_similarity(self(v1), self(v2), dim=-1)
        return {"window": {
            "score_max": diff.max(dim=1).values,
            "score_mean": diff.mean(dim=1),
        }}


def barlow_twins_loss(z1, z2, lambda_bt=5e-3):
    """z1,z2: [B,L,D] -> 展平为 [B*L,D] 后做互相关。"""
    b, length, d = z1.shape
    z1 = z1.reshape(b * length, d)
    z2 = z2.reshape(b * length, d)
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    c = torch.matmul(z1.T, z2) / z1.size(0)
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten().pow_(2).sum()
    return on_diag + lambda_bt * off_diag


def build_model(n_channels, seq_len, data=None):
    return SBT(n_channels, seq_len, D_MODEL, NHEAD, NUM_LAYERS,
               DIM_FF, DROPOUT, LAMBDA_BT, NOISE_STD, MASK_RATIO)


def main():
    run_reconstruction(MODEL_NAME, PREFIX, build_model,
                       description="SBT（SWaT）",
                       batch_size=128, plot_level="window")


if __name__ == "__main__":
    main()

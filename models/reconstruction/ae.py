# -*- coding: utf-8 -*-
"""AE（自编码器）重构异常检测模型。"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch.nn as nn
import torch.nn.functional as F

from models.common.recon_runner import run_reconstruction

MODEL_NAME = "AE"
PREFIX = "ae_swat"


class AutoEncoder(nn.Module):
    """对 seq_len 步窗口做重构：encode 整窗 -> 低维隐变量 -> decode 回整窗。"""

    def __init__(self, seq_len, n_channels, latent_dim=64,
                 hidden_1=512, hidden_2=256, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        in_dim = seq_len * n_channels
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_2, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_2, hidden_1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_1, in_dim))

    def forward(self, x):
        b = x.size(0)
        return self.decoder(self.encoder(x.reshape(b, -1))).view(
            b, self.seq_len, self.n_channels)

    def training_loss(self, x):
        return F.mse_loss(self(x), x)

    def score_batch(self, x):
        recon = self(x)
        diff = recon - x
        abs_err = diff.abs()
        sq_err = diff ** 2
        sq_ts = sq_err.mean(dim=-1)
        return {
            "window": {
                "recon_max": sq_ts.max(dim=1).values,
                "recon_mean": sq_ts.mean(dim=1),
            },
            "point": {
                "point_recon_mae_last": abs_err[:, -1].mean(dim=1),
                "point_recon_mse_last": sq_err[:, -1].mean(dim=1),
            },
            "ts": {
                # 逐时刻的通道平均误差，供 Stage 2 做 window mean/max/top-k
                "abs_ts": abs_err.mean(dim=-1),
                "sq_ts": sq_err.mean(dim=-1),
            },
            "channel": {
                "abs_last": abs_err[:, -1, :],
                "sq_last": sq_err[:, -1, :],
                # 用于逐通道 True vs Recon 对比图
                "recon_last": recon[:, -1, :],
                "true_last": x[:, -1, :],
            },
        }


def build_model(n_channels, seq_len, data=None):
    return AutoEncoder(seq_len, n_channels)


def main():
    run_reconstruction(MODEL_NAME, PREFIX, build_model,
                       description="AE 重构异常检测（SWaT）",
                       save_model=True, plot_level="point",
                       has_channel_metrics=True,
                       plot_reconstruction=True)


if __name__ == "__main__":
    main()

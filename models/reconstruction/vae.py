# -*- coding: utf-8 -*-
"""VAE（变分自编码器）重构异常检测模型。"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.recon_runner import run_reconstruction

MODEL_NAME = "VAE"
PREFIX = "vae_swat"
BETA = 1e-3   # KL 权重


class VAE(nn.Module):
    """对 seq_len 步窗口做变分重构：encode -> (mu, logvar) -> z -> decode。"""

    def __init__(self, seq_len, n_channels, latent_dim=32,
                 hidden_1=512, hidden_2=256, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        in_dim = seq_len * n_channels
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2), nn.GELU(), nn.Dropout(dropout))
        self.fc_mu = nn.Linear(hidden_2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_2, hidden_1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_1, in_dim))

    def encode(self, x):
        h = self.encoder(x.reshape(x.size(0), -1))
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), -10.0, 10.0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        out = self.decoder(z)
        return out.view(x.size(0), self.seq_len, self.n_channels), mu, logvar

    @staticmethod
    def _kl(mu, logvar):
        return -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def training_loss(self, x):
        recon, mu, logvar = self(x)
        return F.mse_loss(recon, x) + BETA * self._kl(mu, logvar).mean()

    def score_batch(self, x):
        recon, mu, logvar = self(x)
        diff = recon - x
        abs_err = diff.abs()
        sq_err = diff ** 2
        kl = self._kl(mu, logvar)
        sq_ts = sq_err.mean(dim=-1)
        return {
            "window": {
                "recon_max": sq_ts.max(dim=1).values,
                "recon_mean": sq_ts.mean(dim=1),
                "kl": kl,
            },
            "point": {
                "point_recon_mae_last": abs_err[:, -1].mean(dim=1),
                "point_recon_mse_last": sq_err[:, -1].mean(dim=1),
                "point_kl": kl,
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
    return VAE(seq_len, n_channels)


def main():
    run_reconstruction(MODEL_NAME, PREFIX, build_model,
                       description="VAE 重构异常检测（SWaT）",
                       save_model=True, plot_level="point",
                       has_channel_metrics=True,
                       plot_reconstruction=True)


if __name__ == "__main__":
    main()

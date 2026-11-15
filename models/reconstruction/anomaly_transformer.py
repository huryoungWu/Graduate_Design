# -*- coding: utf-8 -*-
"""Anomaly Transformer：重构误差 + 关联差异。"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.recon_runner import run_reconstruction

MODEL_NAME = "AnomalyTransformer"
PREFIX = "anomalytransformer_swat"

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 128
DROPOUT = 0.1
KERNEL_SIZE = 3
LAMBDA_KL = 3.0


class AnomalyAttention(nn.Module):
    """带高斯先验关联的注意力。"""

    def __init__(self, win_size, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.win_size = win_size
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.sigma = nn.Parameter(torch.tensor(1.0))

    def _gaussian_prior(self, device):
        idx = torch.arange(self.win_size, device=device).float()
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2
        sigma = torch.clamp(self.sigma, min=0.1)
        prior = torch.exp(-dist / (2 * sigma ** 2))
        prior = prior / prior.sum(dim=-1, keepdim=True)
        return prior.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        b, length, _ = x.shape
        h, d = self.nhead, self.head_dim
        q = self.q_proj(x).view(b, length, h, d).transpose(1, 2)
        k = self.k_proj(x).view(b, length, h, d).transpose(1, 2)
        v = self.v_proj(x).view(b, length, h, d).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d)
        series_assoc = self.dropout(torch.softmax(scores, dim=-1))
        prior = self._gaussian_prior(x.device).expand(b, h, length, length)
        out = torch.matmul(series_assoc, v)
        out = out.transpose(1, 2).contiguous().view(b, length, self.d_model)
        return self.out_proj(out), series_assoc, prior


class AnomalyTransformerBlock(nn.Module):
    def __init__(self, win_size, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.attn = AnomalyAttention(win_size, d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, series_assoc, prior = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, series_assoc, prior


class AnomalyTransformer(nn.Module):
    def __init__(self, n_channels, win_size, d_model=64, nhead=4,
                 num_layers=2, dim_ff=128, dropout=0.1,
                 lambda_kl=3.0):
        super().__init__()
        self.win_size = win_size
        self.n_channels = n_channels
        self.lambda_kl = lambda_kl
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, win_size, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            AnomalyTransformerBlock(win_size, d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.reconstruction = nn.Linear(d_model, n_channels)

    def forward(self, x):
        h = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        series_list, prior_list = [], []
        for blk in self.blocks:
            h, sa, pr = blk(h)
            series_list.append(sa)
            prior_list.append(pr)
        h = self.norm(h)
        return self.reconstruction(h), series_list, prior_list

    def training_loss(self, x):
        recon, series_list, prior_list = self(x)
        rec_loss = F.mse_loss(recon, x)
        assoc = association_discrepancy_loss(series_list, prior_list)
        return rec_loss - self.lambda_kl * assoc

    def score_batch(self, x):
        recon, series_list, prior_list = self(x)
        rec_err = ((recon - x) ** 2).mean(dim=-1)
        rec_abs = (recon - x).abs().mean(dim=-1)
        assoc = 0.0
        for sa, pr in zip(series_list, prior_list):
            kl = (sa * (torch.log(sa + 1e-8) - torch.log(pr + 1e-8))).sum(dim=-1)
            assoc = assoc + kl.mean(dim=1)
        assoc = assoc / len(series_list)
        return {
            "window": {
                "recon_max": rec_err.max(dim=1).values,
                "assoc_max": assoc.max(dim=1).values,
            },
            "point": {
                "point_recon_mae_last": rec_abs[:, -1],
                "point_recon_mse_last": rec_err[:, -1],
                "point_assoc_last": assoc[:, -1],
            },
            "ts": {
                # 逐时刻的通道平均误差，供 Stage 2 做 window mean/max/top-k
                "abs_ts": rec_abs,
                "sq_ts": rec_err,
            },
            "channel": {
                # 用于逐通道 True vs Recon 对比图
                "recon_last": recon[:, -1, :],
                "true_last": x[:, -1, :],
            },
        }

    def make_fused(self, reference, target):
        """用验证集统计量归一化后相加；测试集复用验证集常数。"""
        r = np.asarray(reference["recon_max"])
        a = np.asarray(reference["assoc_max"])
        rmin, rmax = float(r.min()), float(r.max())
        amin, amax = float(a.min()), float(a.max())

        def norm(v, lo, hi):
            return (np.asarray(v) - lo) / (hi - lo + 1e-8)

        out = dict(target)
        out["fused"] = (norm(out["recon_max"], rmin, rmax)
                        + norm(out["assoc_max"], amin, amax))
        return out

    def plot_extra_scores(self, scores):
        if "fused" not in scores:
            return None
        return {"recon": scores["recon_max"],
                "assoc": scores["assoc_max"],
                "fused": scores["fused"]}


def association_discrepancy_loss(series_list, prior_list):
    """KL(prior || series) 的对称化版本。"""
    loss = 0.0
    for sa, pr in zip(series_list, prior_list):
        loss = loss + (pr * (torch.log(pr + 1e-8)
                             - torch.log(sa + 1e-8))).sum(dim=-1).mean()
    return loss / len(series_list)


def build_model(n_channels, seq_len, data=None):
    return AnomalyTransformer(
        n_channels, seq_len, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF,
        DROPOUT, LAMBDA_KL)


def main():
    run_reconstruction(MODEL_NAME, PREFIX, build_model,
                       description="Anomaly Transformer（SWaT）",
                       batch_size=128, save_model=True,
                       plot_level="window",
                       plot_reconstruction=True)


if __name__ == "__main__":
    main()

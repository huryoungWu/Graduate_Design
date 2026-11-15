# -*- coding: utf-8 -*-
"""Graph WaveNet 预测类模型（物理/相关/混合图）。"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.forecast_runner import run_forecasting

GRAPH_MODE = "hybrid"
MODEL_NAME = f"GNN-{GRAPH_MODE}"
PREFIX = f"gnn_swat_{GRAPH_MODE}"


def build_physical_graph(cols):
    """阶段内执行器 -> 传感器，阶段间串联。返回 [C,C] 邻接矩阵。"""
    idx = {c: i for i, c in enumerate(cols)}
    A = np.zeros((len(cols), len(cols)), dtype=np.float32)

    def add_edge(src, dst):
        if src in idx and dst in idx:
            A[idx[src], idx[dst]] = 1.0

    for s in ["P101", "P102", "MV101"]:
        add_edge(s, "FIT101")
    add_edge("FIT101", "LIT101")
    for s in ["P201", "P203", "P204", "P205", "P206", "MV201"]:
        add_edge(s, "FIT201")
    for s in ["AIT201", "AIT202", "AIT203"]:
        add_edge("FIT201", s)
    for s in ["P301", "P302", "MV301", "MV302", "MV303", "MV304"]:
        add_edge(s, "FIT301")
    add_edge("FIT301", "LIT301")
    add_edge("FIT301", "DPIT301")
    for s in ["P402", "P403"]:
        add_edge(s, "FIT401")
    add_edge("FIT401", "LIT401")
    for s in ["AIT401", "AIT402"]:
        add_edge("FIT401", s)
    for s in ["P501"]:
        for d in ["FIT501", "FIT502", "FIT503", "FIT504"]:
            add_edge(s, d)
    for f in ["FIT501", "FIT502", "FIT503", "FIT504"]:
        for p in ["PIT501", "PIT502", "PIT503"]:
            add_edge(f, p)
    for a in ["AIT501", "AIT502", "AIT503", "AIT504"]:
        for f in ["FIT501", "FIT502", "FIT503", "FIT504"]:
            add_edge(a, f)
    add_edge("P602", "FIT601")
    add_edge("FIT101", "FIT201")
    add_edge("FIT201", "FIT301")
    add_edge("FIT301", "DPIT301")
    add_edge("FIT401", "FIT501")
    add_edge("FIT501", "FIT601")
    return A


def build_correlation_graph(train_data, cols, topk=3):
    """用训练段正常数据算相关性，保留每个变量 top-k 邻居。"""
    corr = np.nan_to_num(np.corrcoef(train_data.T), nan=0.0)
    np.fill_diagonal(corr, 0.0)
    A = np.zeros_like(corr)
    for i in range(len(cols)):
        k = min(topk, len(cols) - 1)
        for j in np.argsort(-np.abs(corr[i]))[:k]:
            A[i, j] = abs(corr[i, j])
    return np.maximum(A, A.T).astype(np.float32)


def build_adjacency(cols, train_X, mode=GRAPH_MODE):
    A_phys = build_physical_graph(cols)
    A_corr = build_correlation_graph(train_X, cols, topk=3)
    if mode == "physical":
        A = A_phys
    elif mode == "correlation":
        A = A_corr
    else:
        A = np.maximum(A_phys, 0.5 * A_corr)
    A = torch.tensor(A, dtype=torch.float32)
    A = A + torch.eye(len(cols))
    return A / (A.sum(dim=1, keepdim=True) + 1e-8)


class GraphConv(nn.Module):
    """H' = A_norm @ (H W)"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, A_norm):
        return torch.einsum("ij,bjd->bid", A_norm, self.linear(x))


class GraphWaveNet(nn.Module):
    def __init__(self, n_channels, seq_len, pred_len=1, hidden=64,
                 num_layers=2, dropout=0.1, A_norm=None):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.input_proj = nn.Linear(seq_len, hidden)
        self.gconvs = nn.ModuleList(
            [GraphConv(hidden, hidden) for _ in range(num_layers)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, pred_len)
        self.register_buffer("A_norm", A_norm)

    def forward(self, x):
        h = self.input_proj(x.transpose(1, 2))
        for gconv, norm in zip(self.gconvs, self.norms):
            h = norm(h + self.drop(F.relu(gconv(h, self.A_norm))))
        return self.head(h).transpose(1, 2)


def build_model(n_channels, seq_len, pred_len, data=None):
    if data is None:
        raise ValueError("GraphWaveNet 需要 data 构造邻接矩阵")
    A_norm = build_adjacency(data.sensor_cols, data.train_X, GRAPH_MODE)
    return GraphWaveNet(n_channels, seq_len, pred_len, A_norm=A_norm)


def main():
    run_forecasting(MODEL_NAME, PREFIX, build_model,
                    description=f"Graph WaveNet 预测类模型（{GRAPH_MODE}）",
                    extra_fields={"graph": GRAPH_MODE})


if __name__ == "__main__":
    main()

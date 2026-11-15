# -*- coding: utf-8 -*-
"""
iTransformer 改进变体实验脚本
============================

依据 improve.md 中的六类改进方案，实现以下变体（每个方案一个变体）：

  baseline    : 原始 iTransformer（同管线重跑，用于公平对照）
  graph       : Graph-iTransformer，图结构引导变量注意力
  dual        : Dual-iTransformer，变量注意力 + 因果 TCN 时间分支 + 门控融合
  multiscale  : Multi-scale iTransformer，多尺度下采样嵌入 + 可学习尺度权重
  grouped     : Grouped iTransformer，按 SWaT 工艺段分组注意力 + 跨组交互
  patch       : Patch-iTransformer，变量内部 Patch 建模 + 变量间注意力
  condition   : Condition-aware iTransformer，泵组工况 FiLM 调制
  tcdit       : TC-DiTransformer，improve.md 第 4 节推荐的组合模型
                （图注意力 + TCN 双分支 + 工况 FiLM）

评估协议（见 ad_protocol.py）：
  1. 训练只使用原始 Normal 段（period == "normal"）；校准集从整个 Normal 段
     均匀抽样（避免只取尾部停机段导致大量通道 std≈0），不参与训练。
  2. 原始 Attack 段前 20% 作为验证集，只用于选择评分配置和阈值；
     其余 80% 作为测试集，只报告最终指标。
  3. 主指标为 AUC / PR-AUC（与阈值无关）；F1 采用固定 FPR 口径：
     阈值 = 验证集正常点分数的 (1-target_fpr) 分位，不在测试集上调参。
  4. 评分配置：
       raw_* : 原始绝对残差聚合（mean/max/top3_mean）
       z_*   : 按校准集逐通道残差标准差标准化，并做逐通道裁剪（默认上限 20），
               防止单个通道（如测试期尺度突变的 AIT201）主导聚合
       az_*  : 可选（--roll-window>0），在 z_* 上再做因果滚动鲁棒标准化；
               SWaT 上默认关闭（攻击持续时间较长，滚动基线会吸收攻击信号）
     训练段恒定的通道（SWaT 停机期常见）不参与 z/az 聚合。
  5. 评分配置默认只在 mean 类（raw_mean/z_mean）中选择：max/top3 在存在
     单通道分布漂移时容易与标签反向，可作为诊断指标但不用于选优。
  6. 不再生成逐通道逐日图片（变体数量多，图片量过大），保留指标 CSV。

用法：
  python itransformer_improved.py --variant baseline
  python itransformer_improved.py --variant graph --epochs 10
  python itransformer_improved.py --variant dual --smoke     # 快速自检

常用选项：
  --threshold-mode val_fpr|val|calib   阈值口径（默认 val_fpr，固定 FPR）
  --target-fpr 0.01                    val_fpr 模式的目标误报率
  --roll-window 3600                   滚动自适应窗口秒数，0=关闭
  --calib-mode spread|tail             校准集抽样方式（默认 spread）
  --train-stride 5                     训练窗口抽样，快速实验
  --no-amp                             关闭混合精度
"""

import argparse
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ad_protocol import (
    evaluate_fixed, select_best, select_threshold_fpr,
    rolling_adaptive_normalize, split_train_val_test,
)

# ---- 性能相关后端设置 ----
torch.backends.cudnn.benchmark = True         # 输入形状固定，让 cuDNN 自动选最快卷积算法
torch.backends.cuda.matmul.allow_tf32 = True  # Ampere 显卡上 fp32 矩阵乘走 TF32
torch.backends.cudnn.allow_tf32 = True

# ===================== 1. 配置 =====================
CSV_PATH = r"08_SWaT\SWaT_all_1s.csv"
SEQ_LEN = 60
PRED_LEN = 12
BATCH_SIZE = 512           # 256 -> 512：批次数减半，减少每批固定开销
EPOCHS = 20                # 10 epoch 明显欠拟合（测试 MAE 0.17 vs 0.07），默认改为 20
LR = 1e-4
SEED = 42
CALIB_FRAC = 0.10          # 正常训练窗口中划作残差校准集的比例
CALIB_MODE = "spread"      # "spread"=整个 Normal 段均匀抽样 | "tail"=最后 10%
VAL_FRAC = 0.20            # Attack 段中划作验证集的比例（仅用于选阈值/配置）
THRESH_MODE = "val_fpr"    # "val_fpr"=验证集正常点分位数 | "val"=验证集 max-F1 | "calib"=校准集分位数
TARGET_FPR = 0.01          # THRESH_MODE="val_fpr" 时的目标点级误报率
CALIB_FPR = 0.01           # THRESH_MODE="calib" 时的目标点级误报率
ROLL_WINDOW = 0            # 滚动自适应标准化窗口（秒），0=关闭（SWaT 上不建议开启）
Z_CLIP = 20.0              # 逐通道 z 上限，防止单个通道（如 AIT201）主导聚合；0=不裁剪
MIN_CALIB_STD = 1e-3       # 训练/校准期标准差低于该值的通道不参与 z/az 聚合
TRAIN_STRIDE = 1           # 训练窗口抽样步长：1=不抽样；>1 可成倍减少训练窗口
MEAN_SCORES = ("raw_mean", "z_mean", "az_mean")   # 默认参与选优的稳健聚合

VARIANTS = ["baseline", "graph", "dual", "multiscale",
            "grouped", "patch", "condition", "tcdit"]

torch.manual_seed(SEED)
np.random.seed(SEED)


# ===================== 2. 工具函数 =====================
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stage_of(name):
    """从 SWaT 变量名（如 FIT101 / P602 / MV301）解析工艺段编号 1~6。"""
    m = re.search(r"(\d{3})", name)
    if not m:
        return 0
    return int(m.group(1)[0])


def build_graph_prior(sensor_cols):
    """
    功能/工艺先验图：同工艺段变量相连（1.0），相邻工艺段相连（0.5），
    对角为自环（1.0）。只使用变量命名与工艺顺序，不使用测试数据。
    """
    n = len(sensor_cols)
    stages = np.array([stage_of(c) for c in sensor_cols])
    G = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                G[i, j] = 1.0
            elif stages[i] == stages[j]:
                G[i, j] = 1.0
            elif abs(int(stages[i]) - int(stages[j])) == 1:
                G[i, j] = 0.5
    return torch.tensor(G), stages


# ===================== 3. 读取与预处理 =====================
class WindowDataset(Dataset):
    def __init__(self, data, labels, pump_states, seq_len, pred_len=1):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.pump_states = torch.tensor(pump_states, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        # 只评估预测的第一步，标签必须与这一步严格对齐（point-wise 口径）
        lab = self.labels[idx + self.seq_len]
        # 工况向量：窗口最后一个输入时刻的泵组启停状态（严格因果）
        cond = self.pump_states[idx + self.seq_len - 1]
        return x, y, lab, cond


def prepare_data(args):
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    sensor_cols = [c for c in df.columns if c not in ["timestamp", "label", "period"]]
    df["label"] = df["label"].astype(int)
    df[sensor_cols] = df[sensor_cols].ffill().fillna(0)

    pump_cols = [c for c in sensor_cols if re.fullmatch(r"P\d{3}", c)]
    pump_states = df[pump_cols].values.astype(np.float32)

    # 标准 SWaT 协议：前 495,000 行（Normal 段）训练，Attack 段前 20% 验证，
    # 其余 80% 测试。测试集标签不参与任何选择。
    train_df, val_df, test_df, split_idx, val_end = split_train_val_test(
        df, val_frac=args.val_frac)

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_df[sensor_cols].values)
    val_X = scaler.transform(val_df[sensor_cols].values)
    test_X = scaler.transform(test_df[sensor_cols].values)

    train_label = train_df["label"].values
    val_label = val_df["label"].values
    test_label = test_df["label"].values
    train_pump = pump_states[:split_idx]
    val_pump = pump_states[split_idx:val_end]
    test_pump = pump_states[val_end:]

    train_ds_full = WindowDataset(train_X, train_label, train_pump, SEQ_LEN, PRED_LEN)
    val_ds_full = WindowDataset(val_X, val_label, val_pump, SEQ_LEN, PRED_LEN)
    test_ds_full = WindowDataset(test_X, test_label, test_pump, SEQ_LEN, PRED_LEN)

    # 训练段近似恒定的通道无法可靠标准化，也不应参与 max/top3 聚合
    train_col_std = train_df[sensor_cols].std().values
    channel_valid = train_col_std >= MIN_CALIB_STD

    normal_indices = []
    for i in range(len(train_ds_full)):
        if train_label[i: i + SEQ_LEN + PRED_LEN].max() == 0:
            normal_indices.append(i)

    calib_stride = 1
    if args.smoke:
        train_idx = normal_indices[:20000]
        calib_idx = normal_indices[20000:22000]
        val_ds = Subset(val_ds_full, list(range(min(4000, len(val_ds_full)))))
        test_ds = Subset(test_ds_full, list(range(min(20000, len(test_ds_full)))))
    else:
        if args.calib_mode == "spread":
            # 在整个 Normal 段均匀抽样：SWaT 正常段尾部是停机/低活动期，
            # 只取尾部会让很多通道的残差尺度退化（std≈0），z 分数严重失真。
            calib_stride = max(1, int(round(1.0 / CALIB_FRAC)))
            calib_idx = normal_indices[::calib_stride]
            calib_set = set(calib_idx)
            train_idx = [i for i in normal_indices if i not in calib_set]
        else:  # tail：保持旧行为（最后 10%），仅用于复现旧实验
            n_norm = len(normal_indices)
            n_calib = int(n_norm * CALIB_FRAC)
            train_idx = normal_indices[: n_norm - n_calib]
            calib_idx = normal_indices[n_norm - n_calib:]
        val_ds = val_ds_full
        test_ds = test_ds_full

    # 训练窗口抽样：重叠窗口非常密集，步长 >1 可成倍减少训练样本
    stride = max(1, args.train_stride)
    if stride > 1:
        train_idx = train_idx[::stride]

    print(f"传感器/执行器列数: {len(sensor_cols)} | 泵状态列数: {len(pump_cols)}")
    print(f"训练窗口总数: {len(train_ds_full)} | 正常窗口: {len(normal_indices)}")
    print(f"训练使用: {len(train_idx)} | 校准使用: {len(calib_idx)} | "
          f"验证窗口: {len(val_ds)} | 测试窗口: {len(test_ds)}"
          + (f" | 抽样步长: {stride}" if stride > 1 else ""))
    print(f"被剔除的含攻击训练窗口: {len(train_ds_full) - len(normal_indices)}")
    print(f"协议: train={len(train_df)} 行 | val={len(val_df)} 行 | "
          f"test={len(test_df)} 行 | val/test 边界行号={val_end}")
    print(f"校准集模式: {args.calib_mode}"
          + (f"（全正常段每 {calib_stride} 个窗口取 1 个）"
             if args.calib_mode == "spread" else "（Normal 段最后 10%）"))

    return {
        "sensor_cols": sensor_cols,
        "pump_cols": pump_cols,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "split_idx": split_idx,
        "val_end": val_end,
        "train_ds_full": train_ds_full,
        "train_ds": Subset(train_ds_full, train_idx),
        "calib_ds": Subset(train_ds_full, calib_idx),
        "calib_indices": np.asarray(calib_idx, dtype=np.int64),
        "calib_stride": calib_stride,
        "channel_valid": channel_valid,
        "val_ds": val_ds,
        "test_ds": test_ds,
    }


# ===================== 4. 模型定义 =====================
class ITransformer(nn.Module):
    """原始 iTransformer：变量维 token + 标准 Transformer 编码器。"""

    def __init__(self, n_channels, seq_len, pred_len=1, d_model=128,
                 nhead=8, num_layers=3, dim_ff=256, dropout=0.1,
                 **_ignored):
        super().__init__()
        self.pred_len = pred_len
        self.input_proj = nn.Linear(seq_len, d_model)
        self.var_emb = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x, cond=None):
        h = self.input_proj(x.transpose(1, 2)) + self.var_emb
        h = self.encoder(h)
        h = self.norm(h)
        return self.head(h).transpose(1, 2)


# ---------- 4.1 图结构引导注意力 ----------
class MHABias(nn.Module):
    """多头自注意力，支持在注意力 logits 上叠加先验偏置矩阵。"""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        assert d_model % nhead == 0
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.nhead = nhead
        self.dh = d_model // nhead
        self.dropout = dropout

    def forward(self, x, bias=None):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.nhead, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        if bias is not None:
            att = att + bias
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        out = (att @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out)


class GraphAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MHABias(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias):
        x = x + self.drop(self.attn(self.norm1(x), bias))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class GraphITransformer(nn.Module):
    """A = Softmax(QK^T/sqrt(d) + lambda1 * G_prior + lambda2 * G_learn)"""

    def __init__(self, n_channels, seq_len, pred_len=1, d_model=128,
                 nhead=8, num_layers=3, dim_ff=256, dropout=0.1,
                 graph_prior=None, n_pumps=0, **_ignored):
        super().__init__()
        self.input_proj = nn.Linear(seq_len, d_model)
        self.var_emb = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)
        self.layers = nn.ModuleList([
            GraphAttentionLayer(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)
        self.register_buffer("G_prior", graph_prior)
        self.G_learn = nn.Parameter(torch.zeros(n_channels, n_channels))
        self.lam1 = nn.Parameter(torch.tensor(0.5))
        self.lam2 = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, cond=None):
        h = self.input_proj(x.transpose(1, 2)) + self.var_emb
        bias = (self.lam1 * self.G_prior + self.lam2 * self.G_learn)
        bias = bias.unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            h = layer(h, bias)
        h = self.norm(h)
        return self.head(h).transpose(1, 2)


# ---------- 4.2 双维度特征建模 ----------
class CausalTCN(nn.Module):
    """逐变量因果 TCN，输出最后一个时刻的特征。"""

    def __init__(self, out_dim, hidden=(32, 64, 128), kernel=3, dilations=(1, 2, 4)):
        super().__init__()
        convs = []
        c_in = 1
        for h, d in zip(hidden, dilations):
            convs.append(nn.Conv1d(c_in, h, kernel, dilation=d))
            c_in = h
        self.convs = nn.ModuleList(convs)
        self.kernel = kernel
        self.dilations = dilations
        self.proj = nn.Linear(hidden[-1], out_dim)

    def forward(self, x):  # [N, 1, L]
        for conv, d in zip(self.convs, self.dilations):
            x = F.pad(x, (d * (self.kernel - 1), 0))
            x = F.gelu(conv(x))
        return self.proj(x[..., -1])


class DualITransformer(nn.Module):
    """变量注意力分支 H_v + 时间 TCN 分支 H_t，门控融合。"""

    def __init__(self, n_channels, seq_len, pred_len=1, d_model=128,
                 nhead=8, num_layers=3, dim_ff=256, dropout=0.1,
                 n_pumps=0, **_ignored):
        super().__init__()
        self.input_proj = nn.Linear(seq_len, d_model)
        self.var_emb = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.var_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.tcn = CausalTCN(d_model)
        self.gate = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x, cond=None):
        h_v = self.input_proj(x.transpose(1, 2)) + self.var_emb
        h_v = self.var_encoder(h_v)

        B, L, C = x.shape
        h_t = self.tcn(x.permute(0, 2, 1).reshape(B * C, 1, L)).reshape(B, C, -1)

        alpha = torch.sigmoid(self.gate(torch.cat([h_v, h_t], dim=-1)))
        h = alpha * h_v + (1.0 - alpha) * h_t
        h = self.norm(h)
        return self.head(h).transpose(1, 2)


# ---------- 4.3 多尺度变量嵌入 ----------
class MultiScaleITransformer(nn.Module):
    """多个下采样尺度的窗口分别嵌入，再按可学习权重融合。"""

    def __init__(self, n_channels, seq_len, pred_len=1, d_model=128,
                 nhead=8, num_layers=3, dim_ff=256, dropout=0.1,
                 scales=(1, 2, 4), n_pumps=0, **_ignored):
        super().__init__()
        self.scales = scales
        self.lengths = [max(1, seq_len // s) for s in scales]
        self.projs = nn.ModuleList(
            [nn.Linear(Ls, d_model) for Ls in self.lengths])
        self.scale_logits = nn.Parameter(torch.zeros(len(scales)))
        self.var_emb = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x, cond=None):
        x_t = x.transpose(1, 2)  # [B, C, L]
        weights = torch.softmax(self.scale_logits, dim=0)
        h = 0
        for i, (s, Ls, proj) in enumerate(zip(self.scales, self.lengths, self.projs)):
            xs = x_t if s == 1 else F.adaptive_avg_pool1d(x_t, Ls)
            h = h + weights[i] * proj(xs)
        h = h + self.var_emb
        h = self.encoder(h)
        h = self.norm(h)
        return self.head(h).transpose(1, 2)


# ---------- 4.4 变量分组注意力 ----------
class GroupedITransformer(nn.Module):
    """组内注意力 + 组间注意力，变量分组来自 SWaT 工艺段。"""

    def __init__(self, n_channels, seq_len, pred_len=1, d_model=128,
                 nhead=8, num_layers=3, dim_ff=256, dropout=0.1,
                 group_ids=None, n_groups=0, n_pumps=0, **_ignored):
        super().__init__()
        self.n_groups = n_groups
        active = sorted(torch.unique(group_ids).tolist())
        remap = torch.full((n_groups,), -1, dtype=torch.long)
        for k, g in enumerate(active):
            remap[g] = k
        self.register_buffer("group_ids", group_ids.long())
        self.register_buffer("group_ids_local", remap[group_ids.long()])
        self.register_buffer("active_groups", torch.tensor(active, dtype=torch.long))
        self.n_active = len(active)

        self.input_proj = nn.Linear(seq_len, d_model)
        self.var_emb = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)
        self.group_emb_prior = nn.Parameter(torch.randn(1, n_groups, d_model) * 0.02)
        self.group_emb_cross = nn.Parameter(torch.randn(1, n_groups, d_model) * 0.02)

        intra_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.intra_encoder = nn.TransformerEncoder(
            intra_layer, num_layers=max(1, num_layers - 1))
        cross_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.cross_encoder = nn.TransformerEncoder(cross_layer, num_layers=1)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x, cond=None):
        h0 = self.input_proj(x.transpose(1, 2)) + self.var_emb
        B, C, D = h0.shape

        parts, idx_list = [], []
        for g in self.active_groups.tolist():
            idx = torch.nonzero(self.group_ids == g, as_tuple=False).squeeze(1)
            hg = h0[:, idx, :] + self.group_emb_prior[:, g:g + 1, :]
            if idx.numel() > 1:
                hg = self.intra_encoder(hg)
            parts.append(hg)
            idx_list.append(idx)

        # 还原变量原始顺序
        h_grouped = torch.cat(parts, dim=1)
        perm = torch.cat(idx_list)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        h_intra = h_grouped[:, inv, :]

        # 组间交互
        group_tokens = torch.stack([p.mean(dim=1) for p in parts], dim=1)
        group_tokens = group_tokens + self.group_emb_cross[:, self.active_groups, :]
        cross_tokens = self.cross_encoder(group_tokens)
        h = h_intra + cross_tokens[:, self.group_ids_local, :]
        h = self.norm(h)
        return self.head(h).transpose(1, 2)


# ---------- 4.5 Patch 化变量建模 ----------
class PatchITransformer(nn.Module):
    """变量内部 Patch Attention -> 变量表示 -> 变量间 Attention。"""

    def __init__(self, n_channels, seq_len, pred_len=1, d_model=128,
                 nhead=8, num_layers=3, dim_ff=256, dropout=0.1,
                 patch_len=12, stride=6, n_pumps=0, **_ignored):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.patch_pos = nn.Parameter(
            torch.randn(1, 1, self.num_patches, d_model) * 0.02)
        patch_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.patch_encoder = nn.TransformerEncoder(patch_layer, num_layers=1)

        self.var_emb = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)
        var_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.var_encoder = nn.TransformerEncoder(
            var_layer, num_layers=max(1, num_layers - 1))

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x, cond=None):
        B, L, C = x.shape
        x_t = x.transpose(1, 2)                                   # [B, C, L]
        patches = x_t.unfold(-1, self.patch_len, self.stride)      # [B, C, M, P]
        h = self.patch_proj(patches) + self.patch_pos              # [B, C, M, D]
        M = h.size(2)
        h = self.patch_encoder(h.reshape(B * C, M, -1))            # patch 内注意力
        h = h.mean(dim=1).reshape(B, C, -1)                        # 变量表示
        h = h + self.var_emb
        h = self.var_encoder(h)                                    # 变量间注意力
        h = self.norm(h)
        return self.head(h).transpose(1, 2)


# ---------- 4.6 工况感知建模 ----------
class FiLM(nn.Module):
    """H' = (1 + gamma(C)) * H + beta(C)，零初始化使初始为恒等映射。"""

    def __init__(self, d_model, n_pumps):
        super().__init__()
        self.pump_emb = nn.Embedding(2, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2 * d_model))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h, cond):
        idx = cond.round().long().clamp(0, 1)                 # [B, P]
        c = self.pump_emb(idx).sum(dim=1)                      # [B, D]
        gb = self.mlp(c)
        gamma, beta = gb.chunk(2, dim=-1)
        return (1.0 + gamma).unsqueeze(1) * h + beta.unsqueeze(1)


class ConditionITransformer(nn.Module):
    def __init__(self, n_channels, seq_len, pred_len=1, d_model=128,
                 nhead=8, num_layers=3, dim_ff=256, dropout=0.1,
                 n_pumps=0, **_ignored):
        super().__init__()
        self.input_proj = nn.Linear(seq_len, d_model)
        self.var_emb = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.film = FiLM(d_model, n_pumps)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x, cond=None):
        h = self.input_proj(x.transpose(1, 2)) + self.var_emb
        h = self.encoder(h)
        h = self.norm(h)
        if cond is not None:
            h = self.film(h, cond)
        return self.head(h).transpose(1, 2)


# ---------- 4.7 推荐组合模型 TC-DiTransformer ----------
class TCDiTransformer(nn.Module):
    """拓扑引导变量注意力 + TCN 时间分支 + 门控融合 + 工况 FiLM 调制。"""

    def __init__(self, n_channels, seq_len, pred_len=1, d_model=128,
                 nhead=8, num_layers=3, dim_ff=256, dropout=0.1,
                 graph_prior=None, n_pumps=0, **_ignored):
        super().__init__()
        self.input_proj = nn.Linear(seq_len, d_model)
        self.var_emb = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)
        self.layers = nn.ModuleList([
            GraphAttentionLayer(d_model, nhead, dim_ff, dropout)
            for _ in range(max(1, num_layers - 1))])
        self.register_buffer("G_prior", graph_prior)
        self.G_learn = nn.Parameter(torch.zeros(n_channels, n_channels))
        self.lam1 = nn.Parameter(torch.tensor(0.5))
        self.lam2 = nn.Parameter(torch.tensor(0.2))

        self.tcn = CausalTCN(d_model)
        self.gate = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.film = FiLM(d_model, n_pumps)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x, cond=None):
        h_v = self.input_proj(x.transpose(1, 2)) + self.var_emb
        bias = (self.lam1 * self.G_prior + self.lam2 * self.G_learn)
        bias = bias.unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            h_v = layer(h_v, bias)

        B, L, C = x.shape
        h_t = self.tcn(x.permute(0, 2, 1).reshape(B * C, 1, L)).reshape(B, C, -1)

        alpha = torch.sigmoid(self.gate(torch.cat([h_v, h_t], dim=-1)))
        h = alpha * h_v + (1.0 - alpha) * h_t
        h = self.norm(h)
        if cond is not None:
            h = self.film(h, cond)
        return self.head(h).transpose(1, 2)


def build_model(variant, n_channels, seq_len, group_ids, n_groups,
                graph_prior, n_pumps):
    kw = dict(n_channels=n_channels, seq_len=seq_len, pred_len=PRED_LEN,
              d_model=128, nhead=8, num_layers=3, dim_ff=256, dropout=0.1,
              n_pumps=n_pumps)
    if variant == "baseline":
        return ITransformer(**kw)
    if variant == "graph":
        return GraphITransformer(graph_prior=graph_prior, **kw)
    if variant == "dual":
        return DualITransformer(**kw)
    if variant == "multiscale":
        return MultiScaleITransformer(**kw)
    if variant == "grouped":
        return GroupedITransformer(group_ids=group_ids, n_groups=n_groups, **kw)
    if variant == "patch":
        return PatchITransformer(**kw)
    if variant == "condition":
        return ConditionITransformer(**kw)
    if variant == "tcdit":
        return TCDiTransformer(graph_prior=graph_prior, **kw)
    raise ValueError(f"未知变体: {variant}")


# ===================== 5. 训练与推理 =====================
def train_model(model, train_loader, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log_every = 20                       # 每 20 批才同步一次并刷新进度条

    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        n_seen = 0
        epoch_start = time.time()
        pbar = tqdm(train_loader,
                    desc=f"[{args.variant}] Epoch {epoch:02d}/{args.epochs}",
                    unit="batch", ncols=110, dynamic_ncols=True,
                    mininterval=1.0)
        for step, (x, y, _, cond) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred = model(x, cond)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            if use_amp:
                scaler.unscale_(optimizer)   # 先还原梯度，再做梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach() * x.size(0)
            n_seen += x.size(0)
            if step % log_every == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}",
                                  "avg": f"{float(loss_sum / n_seen):.4f}"})
        pbar.close()
        scheduler.step()
        print(f"[{args.variant}] Epoch {epoch:02d} | Train MSE: "
              f"{float(loss_sum) / max(n_seen, 1):.6f} | "
              f"耗时: {time.time() - epoch_start:.1f}s")
    return time.time() - train_start


@torch.inference_mode()
def collect_preds(model, loader, device):
    model.eval()
    preds, trues, labs = [], [], []
    for x, y, lab, cond in tqdm(loader, desc="Inference", unit="batch",
                                ncols=110, dynamic_ncols=True, leave=False):
        out = model(x.to(device, non_blocking=True),
                    cond.to(device, non_blocking=True))
        preds.append(out[:, 0, :].cpu().numpy())
        trues.append(y[:, 0, :].numpy())
        labs.append(lab.numpy())
    return (np.concatenate(preds, axis=0),
            np.concatenate(trues, axis=0),
            np.concatenate(labs, axis=0))


# ===================== 6. 评估 =====================
def aggregate_scores(residual):
    top3 = np.sort(residual, axis=1)[:, -3:]
    return {
        "mean": residual.mean(axis=1),
        "max": residual.max(axis=1),
        "top3_mean": top3.mean(axis=1),
    }


def build_score_sets(residual, sigma, z_mask, roll_window, z_clip=0.0,
                     calib_stride=1):
    """构造 raw / z / az 三类评分配置。

    raw_* : 原始绝对残差聚合；
    z_*   : 按校准集逐通道残差尺度 sigma 标准化，并裁剪到 [0, z_clip] 后聚合。
            裁剪可防止单个通道（例如测试期尺度突变的 AIT201）长期主导 max/top3；
    az_*  : 可选，在 z_* 上再做因果滚动鲁棒标准化（SWaT 上默认关闭）。
    z_mask 标记训练/校准期尺度可靠的通道，恒定通道不参与 z/az。
    """
    z_res = residual[:, z_mask] / sigma[z_mask]
    if z_clip and z_clip > 0:
        z_res = np.clip(z_res, 0.0, float(z_clip))
    scores = {
        **{f"raw_{k}": v for k, v in aggregate_scores(residual).items()},
        **{f"z_{k}": v for k, v in aggregate_scores(z_res).items()},
    }
    if roll_window and roll_window > 0:
        # 校准分数是按 calib_stride 抽样的，窗口按抽样步长折算到真实秒数
        step = max(1, int(calib_stride))
        win = max(60, int(roll_window) // step)
        min_per = max(30, 120 // step)
        az = rolling_adaptive_normalize(
            {k: v for k, v in scores.items() if k.startswith("z_")},
            window=win, min_periods=min_per)
        scores.update({f"az_{k[2:]}": v for k, v in az.items()})
    return scores


def channel_metrics(true, pred):
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    mask = np.abs(true) > 1e-6
    valid_ratio = mask.mean()
    if mask.sum() > 0:
        mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
    else:
        mape = np.nan
    return mae, rmse, mape, valid_ratio


def main():
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 变体: {args.variant} | epochs: {args.epochs} | "
          f"batch: {args.batch_size} | AMP: {args.amp} | "
          f"train_stride: {args.train_stride} | 阈值模式: {args.threshold_mode} | "
          f"roll_window: {args.roll_window}s")

    data = prepare_data(args)
    sensor_cols = data["sensor_cols"]
    n_channels = len(sensor_cols)

    graph_prior, stages = build_graph_prior(sensor_cols)
    group_values = sorted(set(stages.tolist()))
    group_ids = torch.tensor([group_values.index(s) for s in stages],
                             dtype=torch.long)
    n_groups = len(group_values)

    dl_kwargs = dict(batch_size=args.batch_size, pin_memory=True)
    if args.num_workers > 0:
        dl_kwargs.update(num_workers=args.num_workers,
                         persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(data["train_ds"], shuffle=True, drop_last=True,
                              **dl_kwargs)
    calib_loader = DataLoader(data["calib_ds"], shuffle=False, **dl_kwargs)
    val_loader = DataLoader(data["val_ds"], shuffle=False, **dl_kwargs)
    test_loader = DataLoader(data["test_ds"], shuffle=False, **dl_kwargs)

    model = build_model(args.variant, n_channels, SEQ_LEN, group_ids, n_groups,
                        graph_prior, len(data["pump_cols"])).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    train_time = train_model(model, train_loader, args, device)
    print(f"[{args.variant}] 训练总耗时: {train_time:.1f}s")

    # ---- 校准集：只估计逐通道残差尺度（完全无标签）----
    print("\n===== 校准集推理（正常数据残差尺度）=====")
    c_pred, c_true, _ = collect_preds(model, calib_loader, device)
    c_res = np.abs(c_pred - c_true)
    c_std = c_res.std(axis=0)
    # 训练段恒定（无法可靠标准化）或校准残差过小的通道，z/az 中剔除
    z_mask = data["channel_valid"] & (c_std >= MIN_CALIB_STD)
    if not z_mask.any():
        z_mask[:] = True
    excluded_cols = [sensor_cols[j] for j in np.where(~z_mask)[0]]
    print(f"z 聚合使用通道: {int(z_mask.sum())}/{len(sensor_cols)}"
          + (f" | 校准期恒定已剔除: {', '.join(excluded_cols)}"
             if excluded_cols else ""))
    sigma = np.maximum(c_std, MIN_CALIB_STD)
    calib_scores = build_score_sets(
        c_res, sigma, z_mask, args.roll_window, args.z_clip,
        data["calib_stride"])

    os.makedirs(args.out_dir, exist_ok=True)
    calib_timestamps = data["train_df"]["timestamp"].iloc[
        data["calib_indices"] + SEQ_LEN].astype(str).values
    np.savez_compressed(
        os.path.join(args.out_dir, f"{args.variant}_calib_scores.npz"),
        timestamp=calib_timestamps,
        **{f"score_{k}": v for k, v in calib_scores.items()},
    )

    # ---- 验证集：选阈值 / 评分配置（允许使用验证集标签）----
    print("===== 验证集推理（选阈值/配置）=====")
    v_pred, v_true, v_lab = collect_preds(model, val_loader, device)
    v_res = np.abs(v_pred - v_true)
    val_scores = build_score_sets(v_res, sigma, z_mask, args.roll_window,
                                  args.z_clip)

    # ---- 测试集：只报告最终指标 ----
    print("===== 测试集推理（只报告最终指标）=====")
    pred, true, lab = collect_preds(model, test_loader, device)
    residual = np.abs(pred - true)
    test_scores = build_score_sets(residual, sigma, z_mask, args.roll_window,
                                   args.z_clip)

    # ---- 阈值 / 评分配置选择：只用验证集（或完全无标签的校准集）----
    if args.threshold_mode == "val_fpr":
        # 固定 FPR 口径：阈值 = 验证集正常点分数的 (1-target_fpr) 分位
        thresholds = {name: select_threshold_fpr(score, v_lab, args.target_fpr)[0]
                      for name, score in val_scores.items()}
        threshold_source = f"val_normal_fpr_{args.target_fpr:g}"
    elif args.threshold_mode == "val":
        _, thresholds, _ = select_best(val_scores, v_lab)
        threshold_source = "val_best_f1"
    else:  # calib：校准集正常分数的分位数，不使用任何标签
        thresholds = {name: float(np.quantile(calib_scores[name],
                                              1.0 - args.calib_fpr))
                      for name in val_scores}
        threshold_source = f"calib_normal_fpr_{args.calib_fpr:g}"

    # 评分配置按验证集 PR-AUC（阈值无关）选择，避免 max-F1 对类别比例和阈值抖动敏感。
    # 默认只在稳健的 mean 类中选：测试期出现单通道尺度突变时，max/top3 会与标签反向。
    val_detection = {name: evaluate_fixed(
        name, score, v_lab, thresholds[name], threshold_source=threshold_source)
        for name, score in val_scores.items()}
    if args.select_all_scores:
        select_pool = list(val_detection)
    else:
        select_pool = [k for k in MEAN_SCORES if k in val_detection]
        if not select_pool:
            select_pool = list(val_detection)
    best_name = max(select_pool, key=lambda k: (
        np.isfinite(val_detection[k]["PR_AUC"]), val_detection[k]["PR_AUC"]))
    print(f"\n[{args.variant}] 阈值来源: {threshold_source} | "
          f"选优池: {select_pool} | 选定配置: {best_name} "
          f"(val PR-AUC={val_detection[best_name]['PR_AUC']:.4f}, "
          f"val F1={val_detection[best_name]['F1']:.4f}, "
          f"thr={thresholds[best_name]:.6f})")

    # ---- 预测精度 ----
    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))
    rmse = float(np.sqrt(mse))

    channel_rows, mapes = [], []
    for j, col in enumerate(sensor_cols):
        mae_j, rmse_j, mape_j, vr_j = channel_metrics(true[:, j], pred[:, j])
        channel_rows.append({"channel": col, "MAE": mae_j, "RMSE": rmse_j,
                             "MAPE(%)": mape_j, "valid_ratio": vr_j})
        if np.isfinite(mape_j):
            mapes.append(mape_j)

    print(f"\n[{args.variant}] 预测精度（标准化空间）: MAE {mae:.6f} | "
          f"MSE {mse:.6f} | RMSE {rmse:.6f}")

    # ---- 测试集异常检测：固定阈值，不再用测试标签调参 ----
    detection = []
    for name, score in test_scores.items():
        out = evaluate_fixed(
            name, score, lab, thresholds[name],
            threshold_source=threshold_source,
        )
        out["val_F1"] = val_detection[name]["F1"]
        out["val_PR_AUC"] = val_detection[name]["PR_AUC"]
        out["val_AUC"] = val_detection[name]["AUC"]
        out["val_FPR"] = val_detection[name]["FPR"]
        detection.append(out)
    best = next(d for d in detection if d["name"] == best_name)

    if best["AUC"] < 0.5:
        print(f"  [警告] 选定配置 {best_name} 的测试 AUC={best['AUC']:.4f} < 0.5："
              f"分数与标签反向，通常是某个通道在测试期分布突变所主导，"
              f"可尝试调小 --z-clip 或改用 AUC/PR-AUC 作为主指标。")

    print(f"\n[{args.variant}] 测试集指标（阈值={threshold_source}，测试标签未参与选择）:")
    for d in detection:
        print(f"  {d['name']:>12s} | AUC {d['AUC']:.4f} | PR-AUC {d['PR_AUC']:.4f} | "
              f"P {d['Precision']:.4f} | R {d['Recall']:.4f} | "
              f"F1 {d['F1']:.4f} | FPR {d['FPR']:.4f} | thr {d['threshold']:.6f}")

    # ---- 保存 ----
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, args.variant)

    pd.DataFrame(channel_rows).to_csv(
        f"{prefix}_channel_metrics.csv", index=False, encoding="utf-8-sig")

    pred_timestamps = data["test_df"]["timestamp"].iloc[
        SEQ_LEN: SEQ_LEN + len(pred)].reset_index(drop=True)
    assert len(pred_timestamps) == len(pred)

    score_map = test_scores
    best_score = score_map[best["name"]]
    # 阈值来自验证集（或校准集分位数），不是测试集最优阈值
    pred_anomaly = (best_score >= best["threshold"]).astype(int)
    tp_mask = (pred_anomaly == 1) & (lab == 1)
    fp_mask = (pred_anomaly == 1) & (lab == 0)
    fn_mask = (pred_anomaly == 0) & (lab == 1)

    score_df = pd.DataFrame({
        "timestamp": pred_timestamps.values,
        "label": lab.astype(int),
        "score_raw_mean": test_scores["raw_mean"],
        "score_raw_max": test_scores["raw_max"],
        "score_raw_top3": test_scores["raw_top3_mean"],
        "score_z_mean": test_scores["z_mean"],
        "score_z_max": test_scores["z_max"],
        "score_z_top3": test_scores["z_top3_mean"],
        "pred_anomaly": pred_anomaly,
        "tp": tp_mask.astype(int),
        "fp": fp_mask.astype(int),
        "fn": fn_mask.astype(int),
    })
    if "az_mean" in test_scores:
        score_df["score_az_mean"] = test_scores["az_mean"]
        score_df["score_az_max"] = test_scores["az_max"]
        score_df["score_az_top3"] = test_scores["az_top3_mean"]
    score_df.to_csv(f"{prefix}_anomaly_scores.csv",
                    index=False, encoding="utf-8-sig")

    # 保存验证集分数 / 阈值 / 标签，便于复核阈值来源与 val-test 分布漂移
    val_timestamps = data["val_df"]["timestamp"].iloc[
        SEQ_LEN: SEQ_LEN + len(v_pred)].astype(str).values
    np.savez_compressed(
        os.path.join(out_dir, f"{args.variant}_val_scores.npz"),
        timestamp=val_timestamps,
        labels=v_lab.astype(np.int64),
        **{f"score_{k}": v for k, v in val_scores.items()},
        **{f"threshold_{k}": np.float64(v) for k, v in thresholds.items()},
    )

    extras = {}
    if args.variant in ("graph", "tcdit"):
        extras["lambda1"] = float(model.lam1.item())
        extras["lambda2"] = float(model.lam2.item())
        extras["G_learn_norm"] = float(model.G_learn.norm().item())
    if args.variant == "multiscale":
        extras["scale_weights"] = torch.softmax(
            model.scale_logits.detach(), dim=0).cpu().tolist()
    if args.variant == "grouped":
        extras["n_groups"] = int(n_groups)
        extras["group_values"] = [int(g) for g in group_values]
    if args.variant == "patch":
        extras["patch_len"] = int(model.patch_len)
        extras["stride"] = int(model.stride)
        extras["num_patches"] = int(model.num_patches)

    torch.save(model.state_dict(), f"{prefix}_model.pt")

    metrics = {
        "variant": args.variant,
        "params": int(n_params),
        "train_time_s": float(train_time),
        "epochs": int(args.epochs),
        "seq_len": int(SEQ_LEN),
        "pred_len": int(PRED_LEN),
        "batch_size": int(args.batch_size),
        "amp": bool(args.amp),
        "train_stride": int(args.train_stride),
        "smoke": bool(args.smoke),
        "protocol": {
            "split": "Normal_train / Attack_val / Attack_test",
            "val_frac": float(args.val_frac),
            "calib_mode": args.calib_mode,
            "calib_stride": int(data["calib_stride"]),
            "threshold_mode": args.threshold_mode,
            "threshold_source": threshold_source,
            "target_fpr": float(args.target_fpr),
            "calib_fpr": float(args.calib_fpr),
            "roll_window": int(args.roll_window),
            "z_clip": float(args.z_clip),
            "select_all_scores": bool(args.select_all_scores),
            "selection_criterion": "val_PR_AUC",
            "z_channels_used": int(z_mask.sum()),
            "z_channels_excluded": excluded_cols,
            "train_rows": int(len(data["train_df"])),
            "val_rows": int(len(data["val_df"])),
            "test_rows": int(len(data["test_df"])),
            "val_start": str(data["val_df"]["timestamp"].iloc[0]),
            "val_end": str(data["val_df"]["timestamp"].iloc[-1]),
            "test_start": str(data["test_df"]["timestamp"].iloc[0]),
            "selected_score": best_name,
            "thresholds": {k: float(v) for k, v in thresholds.items()},
            "val_metrics": {
                k: {"AUC": float(d["AUC"]), "PR_AUC": float(d["PR_AUC"]),
                    "Precision": float(d["Precision"]), "Recall": float(d["Recall"]),
                    "F1": float(d["F1"]), "FPR": float(d["FPR"]),
                    "threshold": float(d["threshold"]),
                    "threshold_source": d["threshold_source"]}
                for k, d in val_detection.items()
            },
        },
        "prediction": {"MAE": mae, "MSE": mse, "RMSE": rmse,
                       "MAPE_mean_channels": float(np.mean(mapes)) if mapes else None},
        "detection": detection,
        "best_detection": best,
        "extras": extras,
    }
    with open(f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {prefix}_metrics.json / "
          f"{prefix}_channel_metrics.csv / {prefix}_anomaly_scores.csv / "
          f"{prefix}_model.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--no-amp", action="store_true",
                   help="关闭 CUDA 混合精度（默认开启，可提速约 1.8 倍）")
    p.add_argument("--train-stride", type=int, default=TRAIN_STRIDE,
                   help="训练正常窗口抽样步长，>1 可成倍缩短训练时间（默认 1）")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader 工作进程数（Windows 上默认 0 即可）")
    p.add_argument("--val-frac", type=float, default=VAL_FRAC,
                   help="Attack 段中作为验证集的比例（默认 0.20，仅用于选阈值/配置）")
    p.add_argument("--calib-mode", choices=["spread", "tail"], default=CALIB_MODE,
                   help="spread=整个 Normal 段均匀抽样（默认）；tail=最后 10%（旧行为）")
    p.add_argument("--threshold-mode", choices=["val_fpr", "val", "calib"],
                   default=THRESH_MODE,
                   help="val_fpr=验证集正常点分位数（默认，固定 FPR）；"
                        "val=验证集 max-F1；calib=校准集分位数（无标签）")
    p.add_argument("--target-fpr", type=float, default=TARGET_FPR,
                   help="threshold-mode=val_fpr 时的目标点级误报率（默认 0.01）")
    p.add_argument("--calib-fpr", type=float, default=CALIB_FPR,
                   help="threshold-mode=calib 时的目标点级误报率")
    p.add_argument("--roll-window", type=int, default=ROLL_WINDOW,
                   help="滚动自适应标准化窗口（秒），0=关闭（默认 0；"
                        "SWaT 上开启通常会把持续攻击也归一化掉，不建议）")
    p.add_argument("--z-clip", type=float, default=Z_CLIP,
                   help="逐通道 z 分数裁剪上限，防止单通道主导聚合；0=不裁剪（默认 20）")
    p.add_argument("--select-all-scores", action="store_true",
                   help="评分配置选优时也允许 max/top3；默认只在 mean 类中选择")
    p.add_argument("--out-dir", default="improved_results_v2")
    p.add_argument("--smoke", action="store_true",
                   help="小数据快速自检，默认 1 epoch")
    args = p.parse_args()
    args.amp = not args.no_amp
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.out_dir = os.path.join(args.out_dir, "smoke")
    return args


if __name__ == "__main__":
    main()

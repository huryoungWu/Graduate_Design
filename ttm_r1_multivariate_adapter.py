# -*- coding: utf-8 -*-
"""
TTM-R2 多变量 SWaT 时序预测
============================================================
结构：
    多变量 Input Adapter + 冻结 TTM Backbone + 可训练 Decoder / Forecast Head

数据：
    08_SWaT/SWaT_all_1s.csv
    - 944,919 行，1 秒粒度
    - 45 个过程变量（自动排除 timestamp / label / period）
    - normal 段 495,000 行；attack 段 449,919 行（35 段攻击）

数据划分（严格按时间顺序，避免泄漏）：
    normal 段按 70% / 15% / 15% 划分为
        训练集 / 验证集 / 正常测试集
    attack 段全部作为攻击测试集（分布外测试）
    标准化器只在训练集上拟合

窗口采样：
    训练窗口步长 WINDOW_STRIDE_TRAIN 秒
    评估窗口步长 WINDOW_STRIDE_EVAL  秒
    （1 秒数据逐秒取窗口会产生上百万样本，没有必要）

输入形状：[B, 512, 45]
输出形状：[B, 96, 45]

运行前：
    pip install torch pandas numpy matplotlib scikit-learn transformers
    pip install git+https://github.com/ibm-granite/granite-tsfm.git

运行：
    python ttm_r1_multivariate_adapter.py
    python ttm_r1_multivariate_adapter.py --epochs 1      # 冒烟测试
"""

import os
import json
import time
import random
import argparse
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

# ============================================================
# 0. Transformers 兼容补丁（保留原代码中的处理）
# ============================================================
try:
    from transformers import PreTrainedModel

    _orig_init = PreTrainedModel.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}

    PreTrainedModel.__init__ = _patched_init
except Exception as e:
    print("Transformers 兼容补丁未应用：", repr(e))

from tsfm_public import TinyTimeMixerForPrediction


# ============================================================
# 1. 配置区：通常只需要修改这里
# ============================================================
SEED = 42

DATA_ROOT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"

# SWaT 1 秒数据集
CSV_PATH = os.path.join(DATA_ROOT, "08_SWaT", "SWaT_all_1s.csv")

TIME_COLUMN = "timestamp"
LABEL_COLUMN = "label"        # 0=正常, 1=攻击
PERIOD_COLUMN = "period"      # normal / attack
NORMAL_PERIOD_VALUE = "normal"

# None = 自动使用除 timestamp / label / period 之外的全部数值列（SWaT 为 45 列）。
# 也可以显式填写列表，只训练指定变量。
FEATURES = None

# 本地 TTM-R2 权重目录（不存在时回退到 Hugging Face 模型名）
# granite-timeseries-ttm-r2 的 main 分支 = 512-96-r2（非 ft）
LOCAL_MODEL_DIR = os.path.join(DATA_ROOT, "granite-timeseries-ttm-r2")
MODEL_NAME = "ibm-granite/granite-timeseries-ttm-r2"
MODEL_REVISION = "main"

# TTM-R2 main checkpoint 原生配置为 context=512 / prediction=96，
# 必须与 granite-timeseries-ttm-r2/config.json 保持一致。
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96

# normal 段内部按时间顺序划分
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# 剩余 0.15 作为正常测试集；attack 段单独作为攻击测试集

# 窗口步长（秒）。1 秒数据相邻窗口高度冗余，必须下采样。
WINDOW_STRIDE_TRAIN = 30
WINDOW_STRIDE_EVAL = 60

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE_ADAPTER = 1e-3
LEARNING_RATE_TTM_HEAD = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 10

NUM_WORKERS = 0       # Windows 建议 0
USE_AMP = True        # CUDA 下自动混合精度
GRAD_CLIP = 1.0

# 是否训练 TTM 的 Decoder 和预测 Head。
# True：冻结 Backbone，训练 Adapter + Decoder + Head（推荐）
# False：冻结整个 TTM，只训练 Adapter
TRAIN_TTM_DECODER_HEAD = True

# 用预测残差做逐点异常检测（验证集分位数确定阈值）
RUN_ANOMALY_CHECK = True
DETECTION_QUANTILE = 0.99

# 预测图展示的变量个数：
# None = 每个通道都画；整数 K = 只画训练期标准差最大的前 K 个
PLOT_TOP_K = None

OUTPUT_DIR = os.path.join(DATA_ROOT, "TTM_R2_SWaT_Results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. 随机种子
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. 读取数据
# ============================================================
def read_csv_auto(path):
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"无法读取 CSV：{path}\n最后错误：{last_error}")


def load_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV_PATH 不存在：{CSV_PATH}")

    df = read_csv_auto(CSV_PATH)

    print("\n========== 数据文件 ==========")
    print(CSV_PATH)
    print("原始数据形状：", df.shape)

    # ---------- 时间列 ----------
    if TIME_COLUMN is not None and TIME_COLUMN in df.columns:
        df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
        if df[TIME_COLUMN].isna().any():
            raise ValueError("时间列存在无法解析的值。")
        if not df[TIME_COLUMN].is_monotonic_increasing:
            df = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    timestamps = (
        df[TIME_COLUMN].to_numpy()
        if TIME_COLUMN is not None and TIME_COLUMN in df.columns
        else None
    )

    # ---------- 变量列 ----------
    exclude = {
        c for c in (TIME_COLUMN, LABEL_COLUMN, PERIOD_COLUMN) if c is not None
    }
    if FEATURES is None:
        feature_cols = [c for c in df.columns if c not in exclude]
    else:
        feature_cols = list(FEATURES)

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少变量列：{missing_cols}")

    data = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)

    if data.isna().any().any():
        missing = data.isna().sum()
        raise ValueError(
            "输入变量中存在缺失值或非数值，请先处理后再训练。\n"
            f"各列缺失数量：\n{missing[missing > 0]}"
        )

    values = data.to_numpy(dtype=np.float32)

    # ---------- 标签 / 时期 ----------
    labels = None
    if LABEL_COLUMN is not None and LABEL_COLUMN in df.columns:
        labels_num = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
        if labels_num.isna().any():
            raise ValueError("label 列存在非数值。")
        labels = labels_num.astype(np.int64).to_numpy()

    periods = None
    if PERIOD_COLUMN is not None and PERIOD_COLUMN in df.columns:
        periods = (
            df[PERIOD_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .to_numpy()
        )

    if len(values) <= CONTEXT_LENGTH + PREDICTION_LENGTH:
        raise ValueError(
            f"数据量不足：当前 {len(values)} 行，"
            f"至少需要大于 {CONTEXT_LENGTH + PREDICTION_LENGTH} 行。"
        )

    print("\n========== 数据集信息 ==========")
    print("数据形状：", values.shape)
    print("变量数量：", len(feature_cols))
    print("变量列表：", feature_cols)
    if timestamps is not None:
        print("时间范围：", pd.Timestamp(timestamps[0]), "->",
              pd.Timestamp(timestamps[-1]))
    if labels is not None:
        print("label 分布：", pd.Series(labels).value_counts().to_dict())
    if periods is not None:
        print("period 分布：", pd.Series(periods).value_counts().to_dict())

    return values, feature_cols, labels, periods, timestamps


# ============================================================
# 4. 时间窗口 Dataset
#    按预测目标所在时间段划分，避免随机窗口泄漏
# ============================================================
class WindowDataset(Dataset):
    def __init__(self, scaled_values, target_starts,
                 context_length, prediction_length):
        self.values = scaled_values.astype(np.float32)
        self.target_starts = np.asarray(target_starts, dtype=np.int64)
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.target_starts)

    def __getitem__(self, idx):
        target_start = int(self.target_starts[idx])

        x_start = target_start - self.context_length
        x_end = target_start
        y_end = target_start + self.prediction_length

        x = self.values[x_start:x_end]
        y = self.values[target_start:y_end]

        return (
            torch.from_numpy(x.copy()),
            torch.from_numpy(y.copy()),
            target_start,
        )


def window_starts(first_target, boundary, stride):
    """目标起点 s 满足 s + prediction_length <= boundary。"""
    last_allowed = boundary - PREDICTION_LENGTH
    if last_allowed < first_target:
        return np.empty(0, dtype=np.int64)
    return np.arange(
        first_target, last_allowed + 1, stride, dtype=np.int64
    )


def create_datasets(values, features, periods):
    n = len(values)

    # ---------- 确定 normal 段末尾 ----------
    has_attack = False
    n_normal = n

    if periods is not None:
        is_normal = periods == NORMAL_PERIOD_VALUE
        n_normal = int(is_normal.sum())

        if n_normal == 0:
            raise ValueError("period 列中没有 normal 段。")
        if n_normal < n:
            # SWaT 中 normal 段是连续前缀，attack 段在后；
            # 若顺序不同，按时间窗口预测会产生歧义，这里直接报错。
            if not is_normal[:n_normal].all() or is_normal[n_normal:].any():
                raise ValueError(
                    "normal 段不是文件开头的连续区间，"
                    "当前代码只支持 normal -> attack 的时间顺序。"
                )
            has_attack = True

    # ---------- 划分 normal 段 ----------
    train_end = int(n_normal * TRAIN_RATIO)
    val_end = int(n_normal * (TRAIN_RATIO + VAL_RATIO))

    if train_end < CONTEXT_LENGTH + PREDICTION_LENGTH:
        raise ValueError(
            f"训练段太短：{train_end} 行，至少需要 "
            f"{CONTEXT_LENGTH + PREDICTION_LENGTH} 行。"
        )

    # ---------- 标准化：只在训练段拟合 ----------
    scaler = StandardScaler()
    scaler.fit(values[:train_end])

    # 训练段恒定列：StandardScaler 会把 scale_ 置为 1，
    # 标准化后恒为 0，不会产生 NaN，这里仅打印提示。
    const_mask = np.isclose(scaler.var_, 0.0, atol=1e-12)
    if const_mask.any():
        const_names = [features[i] for i in np.flatnonzero(const_mask)]
        print("\n训练段恒定变量（保留，标准化后恒为 0）：", const_names)

    scaled = scaler.transform(values).astype(np.float32)

    # ---------- 窗口起点 ----------
    train_starts = window_starts(
        CONTEXT_LENGTH, train_end, WINDOW_STRIDE_TRAIN
    )
    val_starts = window_starts(
        train_end, val_end, WINDOW_STRIDE_EVAL
    )
    normal_test_starts = window_starts(
        val_end, n_normal, WINDOW_STRIDE_EVAL
    )
    attack_test_starts = (
        window_starts(n_normal, n, WINDOW_STRIDE_EVAL)
        if has_attack else np.empty(0, dtype=np.int64)
    )

    if len(train_starts) == 0 or len(val_starts) == 0:
        raise ValueError(
            "训练/验证窗口数量为 0。请检查数据长度和划分比例。"
        )
    if len(normal_test_starts) == 0:
        raise ValueError("正常测试窗口数量为 0，请检查划分比例。")

    train_ds = WindowDataset(
        scaled, train_starts, CONTEXT_LENGTH, PREDICTION_LENGTH
    )
    val_ds = WindowDataset(
        scaled, val_starts, CONTEXT_LENGTH, PREDICTION_LENGTH
    )
    normal_test_ds = WindowDataset(
        scaled, normal_test_starts, CONTEXT_LENGTH, PREDICTION_LENGTH
    )
    attack_test_ds = (
        WindowDataset(
            scaled, attack_test_starts, CONTEXT_LENGTH, PREDICTION_LENGTH
        )
        if has_attack else None
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, drop_last=False
    )
    normal_test_loader = DataLoader(
        normal_test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, drop_last=False
    )
    attack_test_loader = (
        DataLoader(
            attack_test_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, drop_last=False
        )
        if attack_test_ds is not None else None
    )

    split_info = {
        "n": n,
        "n_normal": n_normal,
        "train_end": train_end,
        "val_end": val_end,
        "has_attack": has_attack,
    }

    print("\n========== Dataset ==========")
    print(f"总行数：{n} | normal 段：{n_normal} | attack 段：{n - n_normal}")
    print("训练窗口数：", len(train_ds),
          f"（目标区间 [0, {train_end})，步长 {WINDOW_STRIDE_TRAIN}s）")
    print("验证窗口数：", len(val_ds),
          f"（目标区间 [{train_end}, {val_end})）")
    print("正常测试窗口数：", len(normal_test_ds),
          f"（目标区间 [{val_end}, {n_normal})）")
    if attack_test_ds is not None:
        print("攻击测试窗口数：", len(attack_test_ds),
              f"（目标区间 [{n_normal}, {n})）")
    print("输入维度：", len(features))
    print("输出维度：", len(features))

    return (
        train_loader,
        val_loader,
        normal_test_loader,
        attack_test_loader,
        scaler,
        split_info,
    )


# ============================================================
# 5. 多变量输入适配层
#    输入输出形状均为 [B, L, C]
# ============================================================
class MultivariateInputAdapter(nn.Module):
    """
    对每个时刻的 C 个变量做可学习的变量混合。

    x: [B, L, C]
    y: [B, L, C]

    使用零初始化线性映射和残差连接：
        y = x + alpha * Linear(LayerNorm(x))

    初始化时近似恒等映射，不会一开始就破坏预训练模型输入。
    """

    def __init__(self, num_variables, dropout=0.05):
        super().__init__()

        self.norm = nn.LayerNorm(num_variables)
        self.channel_mixing = nn.Linear(
            num_variables, num_variables, bias=True
        )
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        nn.init.zeros_(self.channel_mixing.weight)
        nn.init.zeros_(self.channel_mixing.bias)

    def forward(self, x):
        residual = self.norm(x)
        residual = self.channel_mixing(residual)
        residual = self.dropout(residual)
        return x + self.alpha * residual


# ============================================================
# 6. 组合模型
# ============================================================
class AdaptedTTM(nn.Module):
    def __init__(self, ttm, num_variables):
        super().__init__()

        self.adapter = MultivariateInputAdapter(num_variables)
        self.ttm = ttm

        # 冻结 TTM 全部参数
        for p in self.ttm.parameters():
            p.requires_grad = False

        # 推荐：仅训练 TTM Decoder 和预测 Head
        if TRAIN_TTM_DECODER_HEAD:
            if hasattr(self.ttm, "decoder") and self.ttm.decoder is not None:
                for p in self.ttm.decoder.parameters():
                    p.requires_grad = True

            if hasattr(self.ttm, "head") and self.ttm.head is not None:
                for p in self.ttm.head.parameters():
                    p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)

        # Backbone 始终保持 eval，冻结权重且关闭其 Dropout
        if hasattr(self.ttm, "backbone"):
            self.ttm.backbone.eval()

        # 如果不训练 Decoder/Head，也让整个 TTM 保持 eval
        if not TRAIN_TTM_DECODER_HEAD:
            self.ttm.eval()

        return self

    def forward(self, past_values):
        # [B, 512, C]
        adapted_values = self.adapter(past_values)

        outputs = self.ttm(
            past_values=adapted_values,
            return_loss=False
        )

        # [B, 96, C]
        prediction = outputs.prediction_outputs

        return prediction


# ============================================================
# 7. 加载 TTM-R2
# ============================================================
def load_ttm(num_variables):
    print("\n========== 加载 TTM ==========")

    if os.path.isdir(LOCAL_MODEL_DIR):
        model_source = LOCAL_MODEL_DIR
        print("模型来源：", model_source, "（本地权重）")
        extra_kwargs = {}
    else:
        model_source = MODEL_NAME
        print("模型来源：", model_source, "（Hugging Face）")
        extra_kwargs = {"revision": MODEL_REVISION}

    print("输入变量数量：", num_variables)

    # 关键点：
    # 1. 不传 enable_forecast_channel_mixing。
    # 2. 只覆盖当前版本配置中明确支持的字段。
    # 3. 不强制改 decoder_mode，保留 R2 checkpoint 原始结构，
    #    避免 common_channel -> mix_channel 引起权重形状不匹配。
    # 4. 多变量之间的额外交互由前置 Adapter 学习。
    ttm = TinyTimeMixerForPrediction.from_pretrained(
        model_source,
        num_input_channels=num_variables,
        prediction_channel_indices=list(range(num_variables)),
        **extra_kwargs,
    )

    print("TTM context_length：", ttm.config.context_length)
    print("TTM prediction_length：", ttm.config.prediction_length)
    print("TTM num_input_channels：", ttm.config.num_input_channels)
    print("TTM decoder_mode：", getattr(ttm.config, "decoder_mode", "未配置"))

    if ttm.config.context_length != CONTEXT_LENGTH:
        raise ValueError(
            f"当前 checkpoint context_length={ttm.config.context_length}，"
            f"但代码要求 {CONTEXT_LENGTH}。请同步修改 CONTEXT_LENGTH，"
            "或选择匹配的 TTM checkpoint。"
        )

    if ttm.config.prediction_length < PREDICTION_LENGTH:
        raise ValueError(
            f"模型 prediction_length={ttm.config.prediction_length}，"
            f"小于要求的 {PREDICTION_LENGTH}。"
        )

    if ttm.config.prediction_length != PREDICTION_LENGTH:
        warnings.warn(
            "模型预测长度与代码设置不同。将按模型实际输出长度训练，"
            "请确认数据窗口长度与模型配置一致。"
        )

    return ttm


# ============================================================
# 8. 训练与验证
# ============================================================
def run_epoch(model, loader, optimizer=None, scaler_amp=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0

    for x, y, _ in loader:
        x = x.to(DEVICE, dtype=torch.float32)
        y = y.to(DEVICE, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            amp_enabled = USE_AMP and DEVICE.type == "cuda"

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=amp_enabled
            ):
                pred = model(x)

                # 以模型实际输出长度为准
                target = y[:, :pred.shape[1], :]

                if pred.shape != target.shape:
                    raise RuntimeError(
                        f"预测与标签形状不一致：pred={pred.shape}, "
                        f"target={target.shape}"
                    )

                loss = F.mse_loss(pred.float(), target.float())

            if is_train:
                if scaler_amp is not None and amp_enabled:
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), GRAD_CLIP
                    )
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), GRAD_CLIP
                    )
                    optimizer.step()

        batch_n = x.size(0)
        total_loss += loss.item() * batch_n
        total_count += batch_n

    return total_loss / max(total_count, 1)


def train_model(model, train_loader, val_loader, features):
    adapter_params = list(model.adapter.parameters())

    ttm_trainable_params = [
        p for p in model.ttm.parameters()
        if p.requires_grad
    ]

    param_groups = [
        {
            "params": adapter_params,
            "lr": LEARNING_RATE_ADAPTER
        }
    ]

    if ttm_trainable_params:
        param_groups.append({
            "params": ttm_trainable_params,
            "lr": LEARNING_RATE_TTM_HEAD
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=WEIGHT_DECAY
    )

    amp_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USE_AMP and DEVICE.type == "cuda")
    )

    best_val = float("inf")
    best_epoch = 0
    patience_count = 0

    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_path = os.path.join(OUTPUT_DIR, "best_ttm_swat_adapter.pth")

    print("\n========== 开始训练 ==========")
    print(f"训练窗口数：{len(train_loader.dataset)}，"
          f"batch={BATCH_SIZE}，epochs={EPOCHS}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = run_epoch(
            model, train_loader, optimizer, amp_scaler
        )
        val_loss = run_epoch(
            model, val_loader, optimizer=None
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"{time.time() - t0:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_count = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val,
                "features": features,
                "context_length": CONTEXT_LENGTH,
                "prediction_length": PREDICTION_LENGTH,
            }, best_path)

        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"Early stopping：最佳 Epoch={best_epoch}")
            break

    print(f"\n最佳验证损失：{best_val:.6f}")
    print("最佳模型保存位置：", best_path)

    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "loss_history.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return best_path, history


# ============================================================
# 9. 推理与指标
# ============================================================
@torch.no_grad()
def predict_loader(model, loader):
    model.eval()

    preds = []
    trues = []
    starts = []

    for x, y, start in loader:
        x = x.to(DEVICE, dtype=torch.float32)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(USE_AMP and DEVICE.type == "cuda")
        ):
            pred = model(x)

        pred = pred.float().cpu().numpy()
        y = y.numpy()

        target = y[:, :pred.shape[1], :]

        preds.append(pred)
        trues.append(target)
        starts.append(start.numpy())

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(trues, axis=0),
        np.concatenate(starts, axis=0),
    )


def inverse_transform_3d(data, scaler):
    shape = data.shape
    flat = data.reshape(-1, shape[-1])
    inv = scaler.inverse_transform(flat)
    return inv.reshape(shape)


def calculate_metrics(y_true, y_pred, features):
    rows = []

    for i, name in enumerate(features):
        yt = y_true[:, :, i].reshape(-1).astype(np.float64)
        yp = y_pred[:, :, i].reshape(-1).astype(np.float64)

        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

        std = float(np.std(yt))
        nrmse = rmse / std if std > 1e-12 else np.nan

        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

        # sMAPE 对接近 0 的流量类变量比 MAPE 稳定
        denom = np.abs(yt) + np.abs(yp)
        mask = denom > 1e-8
        smape = (
            200.0 * float(np.mean(np.abs(yt[mask] - yp[mask]) / denom[mask]))
            if mask.any() else np.nan
        )

        rows.append({
            "variable": name,
            "MAE": mae,
            "RMSE": rmse,
            "nRMSE": nrmse,
            "R2": r2,
            "sMAPE_percent": smape,
        })

    return pd.DataFrame(rows)


def persistence_baseline(raw_values, starts, true, pred, features):
    """
    持续性基线：把上下文最后一个观测值重复 prediction_length 次。

    对 1 秒级数据这是很强的基线，必须与 TTM 对比才有意义。
    """
    horizon = true.shape[1]
    last_value = raw_values[starts - 1][:, None, :].astype(np.float64)
    pers = np.repeat(last_value, horizon, axis=1)

    pers_err = true.astype(np.float64) - pers
    model_err = true.astype(np.float64) - pred.astype(np.float64)

    pers_mae = np.mean(np.abs(pers_err), axis=(0, 1))
    pers_rmse = np.sqrt(np.mean(pers_err ** 2, axis=(0, 1)))
    ttm_mae = np.mean(np.abs(model_err), axis=(0, 1))
    ttm_rmse = np.sqrt(np.mean(model_err ** 2, axis=(0, 1)))

    pers_mse = np.mean(pers_err ** 2, axis=(0, 1))
    ttm_mse = np.mean(model_err ** 2, axis=(0, 1))

    # 若某变量在该测试段完全恒定，持续性基线误差为 0，
    # skill 无定义（不参与统计），避免除零放大。
    skill = np.full(len(features), np.nan, dtype=np.float64)
    valid = pers_mse > 1e-12
    skill[valid] = 1.0 - ttm_mse[valid] / pers_mse[valid]

    return pd.DataFrame({
        "variable": features,
        "persistence_MAE": pers_mae,
        "persistence_RMSE": pers_rmse,
        "TTM_MAE": ttm_mae,
        "TTM_RMSE": ttm_rmse,
        "skill_vs_persistence": skill,
    })


def evaluate_set(model, loader, scaler, features, set_name, raw_values):
    pred_scaled, true_scaled, starts = predict_loader(model, loader)

    pred = inverse_transform_3d(pred_scaled, scaler)
    true = inverse_transform_3d(true_scaled, scaler)

    metrics = calculate_metrics(true, pred, features)
    scaled_mse = float(np.mean((pred_scaled - true_scaled) ** 2))

    baseline = persistence_baseline(
        raw_values, starts, true, pred, features
    )
    baseline.to_csv(
        os.path.join(OUTPUT_DIR, f"baseline_persistence_{set_name}.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    skill_values = baseline["skill_vs_persistence"].dropna()
    skill_median = (
        float(np.median(skill_values)) if len(skill_values) else np.nan
    )
    skill_positive = int((skill_values > 0).sum())

    print(f"\n========== 测试集指标：{set_name} ==========")
    print(metrics.to_string(index=False))
    print(f"标准化空间 MSE：{scaled_mse:.6f}")
    print(f"持续性基线 macro RMSE："
          f"{baseline['persistence_RMSE'].mean():.4f} | "
          f"TTM macro RMSE：{baseline['TTM_RMSE'].mean():.4f}")
    print(f"TTM 相对基线的 median skill：{skill_median:.4f} | "
          f"优于基线的变量：{skill_positive}/{len(skill_values)}")

    metrics.to_csv(
        os.path.join(OUTPUT_DIR, f"metrics_{set_name}.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f"predictions_{set_name}.npz"),
        pred=pred.astype(np.float16),
        true=true.astype(np.float16),
        starts=starts.astype(np.int64),
    )

    summary = {
        "dataset": set_name,
        "windows": int(len(starts)),
        "scaled_MSE": scaled_mse,
        "MAE_mean": float(metrics["MAE"].mean()),
        "RMSE_mean": float(metrics["RMSE"].mean()),
        "nRMSE_median": float(np.nanmedian(metrics["nRMSE"])),
        "R2_median": float(np.nanmedian(metrics["R2"])),
        "sMAPE_median": float(np.nanmedian(metrics["sMAPE_percent"])),
        "persistence_RMSE_mean": float(
            baseline["persistence_RMSE"].mean()
        ),
        "skill_vs_persistence_median": skill_median,
        "skill_positive_vars": skill_positive,
        "skill_evaluable_vars": int(len(skill_values)),
    }

    return {
        "metrics": metrics,
        "summary": summary,
        "pred_scaled": pred_scaled,
        "true_scaled": true_scaled,
        "starts": starts,
        "pred": pred,
        "true": true,
    }


# ============================================================
# 10. 逐时间步预测误差（用于异常检测评估）
# ============================================================
def compute_channel_step_errors(pred_scaled, true_scaled, starts, lo, hi):
    """
    将每个预测窗口的逐时刻、逐通道平方误差累加到全局时间轴的 [lo, hi)。

    返回：
        err: [hi-lo, C]，未被任何窗口覆盖的时刻为 NaN
        cnt: [hi-lo]，覆盖该时刻的窗口数
    """
    horizon = pred_scaled.shape[1]
    num_channels = pred_scaled.shape[2]

    err_sum = np.zeros((hi - lo, num_channels), dtype=np.float64)
    err_cnt = np.zeros(hi - lo, dtype=np.float64)

    for i, s in enumerate(starts):
        s = int(s)
        a = max(s, lo)
        e = min(s + horizon, hi)
        if a >= e:
            continue
        sq = (
            pred_scaled[i, a - s:e - s].astype(np.float64)
            - true_scaled[i, a - s:e - s].astype(np.float64)
        ) ** 2
        err_sum[a - lo:e - lo] += sq
        err_cnt[a - lo:e - lo] += 1

    err = np.full((hi - lo, num_channels), np.nan, dtype=np.float64)
    valid = err_cnt > 0
    err[valid] = err_sum[valid] / err_cnt[valid, None]

    return err, err_cnt


def normalized_detection_score(err, cal_mean):
    """
    归一化残差：
        每个通道的误差除以其在验证集上的平均误差，再对通道取平均。

    这样可避免高噪声通道（流量等）淹没低噪声通道（液位、泵状态）的异常信号。
    """
    return np.mean(err / cal_mean[None, :], axis=1)


def detection_metrics(err, labels, threshold):
    valid = np.isfinite(err)
    pred_pos = (err > threshold) & valid
    true_pos = (labels == 1) & valid

    tp = int(np.sum(pred_pos & true_pos))
    fp = int(np.sum(pred_pos & ~true_pos))
    fn = int(np.sum(~pred_pos & true_pos))
    tn = int(np.sum(~pred_pos & ~true_pos))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
    }


def run_anomaly_check(model, val_loader, normal_loader, attack_loader,
                      labels, split_info, timestamps):
    print("\n========== 残差异常检测评估 ==========")

    n = split_info["n"]
    n_normal = split_info["n_normal"]
    train_end = split_info["train_end"]
    val_end = split_info["val_end"]

    # ---------- 验证集逐通道误差 → 通道基线 + 阈值 ----------
    val_pred, val_true, val_starts = predict_loader(model, val_loader)
    val_err, _ = compute_channel_step_errors(
        val_pred, val_true, val_starts, train_end, val_end
    )

    cal_mean = np.nanmean(val_err, axis=0)
    cal_mean = np.maximum(cal_mean, 1e-8)

    val_score = normalized_detection_score(val_err, cal_mean)
    val_finite = val_score[np.isfinite(val_score)]
    threshold = float(np.quantile(val_finite, DETECTION_QUANTILE))

    # ---------- 攻击测试集 ----------
    attack_pred, attack_true, attack_starts = predict_loader(
        model, attack_loader
    )
    attack_err, _ = compute_channel_step_errors(
        attack_pred, attack_true, attack_starts, n_normal, n
    )
    attack_score = normalized_detection_score(attack_err, cal_mean)

    labels_test = labels[n_normal:]
    raw_err = np.nanmean(attack_err, axis=1)

    mean_raw_normal = float(np.nanmean(raw_err[labels_test == 0]))
    mean_raw_attack = float(np.nanmean(raw_err[labels_test == 1]))
    mean_score_normal = float(
        np.nanmean(attack_score[labels_test == 0])
    )
    mean_score_attack = float(
        np.nanmean(attack_score[labels_test == 1])
    )
    ratio = (
        mean_score_attack / mean_score_normal
        if mean_score_normal > 1e-12 else float("inf")
    )

    print(f"验证集 {DETECTION_QUANTILE:.1%} 分位阈值：{threshold:.6f}")
    print(f"攻击测试段（原始 MSE）：正常点={mean_raw_normal:.6f} | "
          f"攻击点={mean_raw_attack:.6f}")
    print(f"攻击测试段（归一化分数）：正常点={mean_score_normal:.6f} | "
          f"攻击点={mean_score_attack:.6f} | 倍数={ratio:.2f}x")

    # ---------- 阈值扫描 ----------
    sweep = []
    for q in [0.95, 0.98, 0.99, 0.995, 0.999]:
        thr = float(np.quantile(val_finite, q))
        m = detection_metrics(attack_score, labels_test, thr)
        m["quantile"] = q
        m["threshold"] = thr
        sweep.append(m)
        print(f"  q={q:<6} thr={thr:.6f} | "
              f"P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} FPR={m['fpr']:.3f}")

    primary = detection_metrics(
        attack_score, labels_test, threshold
    )

    # ---------- 正常测试集误报率 ----------
    normal_pred, normal_true, normal_starts = predict_loader(
        model, normal_loader
    )
    normal_err, _ = compute_channel_step_errors(
        normal_pred, normal_true, normal_starts, val_end, n_normal
    )
    normal_score = normalized_detection_score(normal_err, cal_mean)
    normal_score = normal_score[np.isfinite(normal_score)]
    normal_fpr = float(np.mean(normal_score > threshold))

    print(f"正常测试集误报率（阈值来自验证集）：{normal_fpr:.4f}")

    result = {
        "score_type": (
            "mean over channels of "
            "(step squared error / validation mean step squared error)"
        ),
        "detection_quantile": DETECTION_QUANTILE,
        "threshold": threshold,
        "test_normal_points": int(np.sum(labels_test == 0)),
        "test_attack_points": int(np.sum(labels_test == 1)),
        "mean_raw_error_normal": mean_raw_normal,
        "mean_raw_error_attack": mean_raw_attack,
        "mean_score_normal": mean_score_normal,
        "mean_score_attack": mean_score_attack,
        "score_ratio_attack_vs_normal": ratio,
        "normal_test_fpr": normal_fpr,
        "primary": primary,
        "threshold_sweep": sweep,
        "channel_calibration_mean": cal_mean.tolist(),
    }

    with open(
        os.path.join(OUTPUT_DIR, "anomaly_detection_metrics.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    np.save(
        os.path.join(OUTPUT_DIR, "attack_anomaly_scores.npy"),
        attack_score
    )

    plot_residual_timeline(
        timestamps, attack_score, labels, n_normal, threshold
    )

    return result


# ============================================================
# 11. 可视化
# ============================================================
def plot_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (standardized space)")
    plt.title("TTM + Multivariate Adapter - SWaT")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "loss_curve.png"),
        dpi=200
    )
    plt.close()


def _format_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")


def plot_forecasts(result_map, features, timestamps, labels,
                   n_normal, plot_indices):
    horizon = PREDICTION_LENGTH

    for idx in plot_indices:
        name = features[idx]
        fig, axes = plt.subplots(2, 1, figsize=(12, 7))

        for ax, set_name in zip(axes, ["normal_test", "attack_test"]):
            if set_name not in result_map:
                ax.axis("off")
                continue

            res = result_map[set_name]
            starts = res["starts"]
            pred = res["pred"]
            true = res["true"]

            if set_name == "normal_test":
                w = len(starts) // 2
            else:
                attack_idx = (
                    int(np.flatnonzero(labels == 1)[0])
                    if labels is not None and (labels == 1).any()
                    else n_normal
                )
                w = int(
                    np.searchsorted(starts, attack_idx, side="right") - 1
                )
                w = int(np.clip(w, 0, len(starts) - 1))

            s = int(starts[w])
            if timestamps is not None:
                x_axis = pd.to_datetime(
                    timestamps[s:s + horizon]
                )
                ax.plot(x_axis, true[w, :, idx],
                        label="Actual", linewidth=1.6)
                ax.plot(x_axis, pred[w, :, idx],
                        label="Forecast", linewidth=1.6)
                _format_time_axis(ax)
            else:
                ax.plot(true[w, :, idx],
                        label="Actual", linewidth=1.6)
                ax.plot(pred[w, :, idx],
                        label="Forecast", linewidth=1.6)

            ax.set_title(
                f"{set_name} | target starts at "
                f"{pd.Timestamp(timestamps[s]) if timestamps is not None else s}"
            )
            ax.set_ylabel(name)
            ax.grid(alpha=0.25)
            ax.legend(loc="best")

        fig.suptitle(f"TTM SWaT Forecast - {name}")
        fig.tight_layout()
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"forecast_{name}.png"),
            dpi=200
        )
        plt.close(fig)


def plot_residual_timeline(timestamps, score, labels,
                           n_normal, threshold):
    """score 已是对齐到 attack 测试段（从 n_normal 开始）的数组。"""
    if timestamps is None or labels is None:
        return

    ts = pd.to_datetime(timestamps[n_normal:])
    e = np.asarray(score)
    lab = labels[n_normal:]

    if len(e) != len(ts):
        raise ValueError(
            f"残差与时间轴长度不一致：{len(e)} vs {len(ts)}"
        )

    step = max(1, len(ts) // 20000)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(ts[::step], e[::step], linewidth=0.5,
            color="tab:blue", label="Normalized forecast residual")
    ax.axhline(threshold, color="tab:red", linestyle="--",
               linewidth=1.2, label=f"Threshold (val q={DETECTION_QUANTILE})")

    attack_idx = np.flatnonzero(lab == 1)
    if attack_idx.size:
        runs = np.split(
            attack_idx, np.where(np.diff(attack_idx) > 1)[0] + 1
        )
        for r in runs:
            ax.axvspan(
                ts[r[0]], ts[r[-1]],
                color="tab:red", alpha=0.12, linewidth=0
            )

    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized residual")
    ax.set_title(
        "SWaT attack period - per-timestep normalized forecast residual "
        "(red bands = attack labels)"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    _format_time_axis(ax)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "residual_timeline_attack.png"),
        dpi=200
    )
    plt.close(fig)


# ============================================================
# 12. 主函数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="TTM-R2 SWaT multivariate forecasting"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="指定已有 checkpoint 路径，跳过训练直接推理"
    )
    parser.add_argument(
        "--no-anomaly-check", action="store_true",
        help="只做预测训练/推理，不计算残差异常检测指标"
    )
    return parser.parse_args()


def main():
    global EPOCHS, BATCH_SIZE, OUTPUT_DIR, RUN_ANOMALY_CHECK

    args = parse_args()
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    OUTPUT_DIR = args.output_dir
    RUN_ANOMALY_CHECK = not args.no_anomaly_check
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    set_seed(SEED)

    print("运行设备：", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU：", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # ---------- 读取数据 ----------
    values, features, labels, periods, timestamps = load_data()

    # ---------- 构建 Dataset ----------
    (
        train_loader,
        val_loader,
        normal_test_loader,
        attack_test_loader,
        data_scaler,
        split_info,
    ) = create_datasets(values, features, periods)

    if timestamps is not None:
        print("\n========== 时间段 ==========")
        print("训练段：", pd.Timestamp(timestamps[0]), "->",
              pd.Timestamp(timestamps[split_info["train_end"] - 1]))
        print("验证段：", pd.Timestamp(timestamps[split_info["train_end"]]),
              "->", pd.Timestamp(timestamps[split_info["val_end"] - 1]))
        print("正常测试段：",
              pd.Timestamp(timestamps[split_info["val_end"]]), "->",
              pd.Timestamp(timestamps[split_info["n_normal"] - 1]))
        if split_info["has_attack"]:
            print("攻击测试段：",
                  pd.Timestamp(timestamps[split_info["n_normal"]]), "->",
                  pd.Timestamp(timestamps[-1]))

    # ---------- 加载 TTM ----------
    ttm = load_ttm(len(features))

    # ---------- 组装模型 ----------
    model = AdaptedTTM(
        ttm=ttm,
        num_variables=len(features)
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\n========== 模型参数 ==========")
    print(f"总参数量：{total_params:,}")
    print(f"可训练参数量：{trainable_params:,}")
    print(f"可训练比例：{100 * trainable_params / total_params:.4f}%")

    print("\n========== 冻结情况 ==========")
    print("Backbone 可训练参数：",
          sum(p.numel() for p in model.ttm.backbone.parameters()
              if p.requires_grad))

    if hasattr(model.ttm, "decoder") and model.ttm.decoder is not None:
        print("Decoder 可训练参数：",
              sum(p.numel() for p in model.ttm.decoder.parameters()
                  if p.requires_grad))

    print("Head 可训练参数：",
          sum(p.numel() for p in model.ttm.head.parameters()
              if p.requires_grad))

    # ---------- 训练 / 加载已有 checkpoint ----------
    history = None

    if args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(
                f"checkpoint 不存在：{args.checkpoint}"
            )

        checkpoint = torch.load(
            args.checkpoint,
            map_location=DEVICE,
            weights_only=False
        )
        ckpt_features = checkpoint.get("features")
        if ckpt_features is not None and list(ckpt_features) != list(features):
            raise ValueError(
                "checkpoint 的变量列表与当前数据不一致：\n"
                f"checkpoint={ckpt_features}\n当前={features}"
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        print("\n已加载 checkpoint，跳过训练：", args.checkpoint)
        print(f"  Epoch={checkpoint.get('epoch', '?')} | "
              f"Val Loss={checkpoint.get('best_val_loss', float('nan')):.6f}")

        hist_path = os.path.join(OUTPUT_DIR, "loss_history.csv")
        if os.path.exists(hist_path):
            hist_df = pd.read_csv(hist_path)
            history = {
                c: hist_df[c].tolist() for c in hist_df.columns
            }

    else:
        best_path, history = train_model(
            model, train_loader, val_loader, features
        )

        checkpoint = torch.load(
            best_path,
            map_location=DEVICE,
            weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"已加载最佳模型（Epoch {checkpoint['epoch']}，"
              f"Val Loss {checkpoint['best_val_loss']:.6f}）")

    # ---------- 测试集推理 ----------
    result_map = {}

    result_map["normal_test"] = evaluate_set(
        model, normal_test_loader, data_scaler,
        features, "normal_test", values
    )

    if attack_test_loader is not None:
        result_map["attack_test"] = evaluate_set(
            model, attack_test_loader, data_scaler,
            features, "attack_test", values
        )

    # ---------- 汇总指标 ----------
    summary_rows = [
        res["summary"] for res in result_map.values()
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        os.path.join(OUTPUT_DIR, "metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== 指标汇总 ==========")
    print(summary.to_string(index=False))

    # ---------- 残差异常检测 ----------
    if (
        RUN_ANOMALY_CHECK
        and attack_test_loader is not None
        and labels is not None
    ):
        run_anomaly_check(
            model,
            val_loader,
            normal_test_loader,
            attack_test_loader,
            labels,
            split_info,
            timestamps,
        )

    # ---------- 可视化 ----------
    if history is not None:
        plot_loss(history)

    train_std = np.std(
        values[:split_info["train_end"]], axis=0
    )
    if PLOT_TOP_K is None or PLOT_TOP_K >= len(features):
        plot_indices = np.arange(len(features))
    else:
        plot_indices = np.argsort(train_std)[::-1][:PLOT_TOP_K]

    plot_forecasts(
        result_map,
        features,
        timestamps,
        labels,
        split_info["n_normal"],
        plot_indices,
    )

    # ---------- 完成 ----------
    print("\n========== 完成 ==========")
    print("结果目录：", OUTPUT_DIR)
    print("包含：")
    if args.checkpoint is None:
        print("  best_ttm_swat_adapter.pth")
        print("  loss_history.csv / loss_curve.png")
    print("  metrics_normal_test.csv")
    print("  baseline_persistence_normal_test.csv")
    if attack_test_loader is not None:
        print("  metrics_attack_test.csv")
        print("  baseline_persistence_attack_test.csv")
    print("  metrics_summary.csv")
    print("  predictions_normal_test.npz")
    if attack_test_loader is not None:
        print("  predictions_attack_test.npz")
    if RUN_ANOMALY_CHECK and attack_test_loader is not None:
        print("  anomaly_detection_metrics.json")
        print("  residual_timeline_attack.png")
        print("  attack_anomaly_scores.npy")
    print(f"  每个变量的预测图（共 {len(plot_indices)} 个通道）")


if __name__ == "__main__":
    main()

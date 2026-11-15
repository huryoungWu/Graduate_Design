# -*- coding: utf-8 -*-
"""SWaT 数据读取、Normal/Attack 切分、标准化与窗口采样。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from .config import RunConfig
from .protocol import split_train_val_test

EXCLUDE = ["timestamp", "label", "period"]


@dataclass
class DataBundle:
    sensor_cols: list[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_X: np.ndarray          # 标准化后的输入
    val_X: np.ndarray
    test_X: np.ndarray
    train_label: np.ndarray
    val_label: np.ndarray
    test_label: np.ndarray
    scaler: object              # 训练段拟合的 scaler（Standard/MinMax），用于反变换
    split_idx: int
    val_end: int

    @property
    def n_channels(self) -> int:
        return len(self.sensor_cols)


def build_scaler(kind: str = "standard"):
    """按名称创建 scaler；minmax 用于 USAD 等 sigmoid 输出的模型。"""
    if kind in (None, "", "standard"):
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "minmax11":
        return MinMaxScaler(feature_range=(-1, 1))
    raise ValueError(f"未知 scaler: {kind}")


def scaler_stats(scaler) -> tuple[np.ndarray, np.ndarray]:
    """返回可写入 meta 的 (offset, scale)。

    - StandardScaler: (mean_, scale_)
    - MinMaxScaler:   (min_,   scale_)，其中 scale_ = max - min
    """
    if isinstance(scaler, MinMaxScaler):
        return (np.asarray(scaler.min_, dtype=np.float64),
                np.asarray(scaler.scale_, dtype=np.float64))
    if isinstance(scaler, StandardScaler):
        return (np.asarray(scaler.mean_, dtype=np.float64),
                np.asarray(scaler.scale_, dtype=np.float64))
    raise TypeError(f"不支持的 scaler 类型: {type(scaler)}")


def load_swat(csv_path: str | Path, val_frac: float = 0.20,
              verbose: bool = True, scaler: str = "standard") -> DataBundle:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    sensor_cols = [c for c in df.columns if c not in EXCLUDE]
    if not sensor_cols:
        raise ValueError("未找到传感器/执行器列")
    df["label"] = df["label"].astype(int)
    df[sensor_cols] = df[sensor_cols].ffill().fillna(0)

    train_df, val_df, test_df, split_idx, val_end = split_train_val_test(
        df, val_frac=val_frac)
    del df  # 原始整表不再需要；WADI 上可省约 0.8 GB

    scaler_obj = build_scaler(scaler)
    # 模型输入最终都会被 Dataset 转成 float32；这里直接转 float32，
    # 避免 float64 数组与 tensor 同时驻留（WADI 上省约 0.8 GB）。
    train_X = scaler_obj.fit_transform(
        train_df[sensor_cols].values).astype(np.float32, copy=False)
    val_X = scaler_obj.transform(
        val_df[sensor_cols].values).astype(np.float32, copy=False)
    test_X = scaler_obj.transform(
        test_df[sensor_cols].values).astype(np.float32, copy=False)

    if verbose:
        print(f"[data] 传感器/执行器列数={len(sensor_cols)} | scaler={scaler}")
        print(f"[data] train={len(train_df)} 行 | val={len(val_df)} 行 | "
              f"test={len(test_df)} 行")
        print(f"[data] val 时间: {val_df['timestamp'].iloc[0]} ~ "
              f"{val_df['timestamp'].iloc[-1]}")
        print(f"[data] test 时间: {test_df['timestamp'].iloc[0]} ~ "
              f"{test_df['timestamp'].iloc[-1]}")

    return DataBundle(
        sensor_cols=sensor_cols,
        train_df=train_df, val_df=val_df, test_df=test_df,
        train_X=train_X, val_X=val_X, test_X=test_X,
        train_label=train_df["label"].values,
        val_label=val_df["label"].values,
        test_label=test_df["label"].values,
        scaler=scaler_obj,
        split_idx=split_idx, val_end=val_end,
    )


class ForecastDataset(Dataset):
    """预测类窗口：x=[seq_len,C]，y 从 t+horizon 开始、长度 pred_len。

    ``horizon=1`` 即原来的下一时刻预测；``horizon=60`` 表示用截止 t 的
    输入窗口预测 t+60 起的目标（严格因果，输入不含目标时刻）。
    标签返回预测窗口内**每一步**的标签（形状 ``[pred_len]``），
    这样多步评估时每一步都能找到对应的真实标签。
    """

    def __init__(self, data, labels, seq_len: int, pred_len: int = 1,
                 horizon: int = 1):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.horizon = max(1, int(horizon))
        self.n = max(0, len(data) - self.seq_len - self.horizon
                     - self.pred_len + 2)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        start = idx + self.seq_len + self.horizon - 1
        x = self.data[idx: idx + self.seq_len]
        y = self.data[start: start + self.pred_len]
        lab = self.labels[start: start + self.pred_len]
        return x, y, lab


class ReconstructionDataset(Dataset):
    """重构类窗口：x=[seq_len,C]，标签为窗口内是否含攻击。"""

    def __init__(self, data, labels, seq_len: int):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.seq_len = int(seq_len)
        self.n = len(data) - self.seq_len + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.seq_len]
        lab = self.labels[idx: idx + self.seq_len].max()
        return x, lab


def _normal_indices(labels: np.ndarray, window: int) -> np.ndarray:
    """选出窗口内全为正常的起点（训练集只使用这些窗口）。"""
    windows = np.lib.stride_tricks.sliding_window_view(labels, window)
    return np.flatnonzero(windows.max(axis=1) == 0)


def _make_loader(dataset, cfg: RunConfig, shuffle: bool) -> DataLoader:
    # 训练用独立的随机数生成器：样本顺序只由 cfg.seed 决定，
    # 不受模型初始化消耗随机数的影响（不同模型实现也能用相同顺序）。
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=shuffle and len(dataset) >= cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_forecast_loaders(bundle: DataBundle, cfg: RunConfig):
    """返回 train/val/test DataLoader；训练只使用正常窗口。"""
    train_full = ForecastDataset(
        bundle.train_X, bundle.train_label, cfg.seq_len, cfg.pred_len,
        cfg.horizon)
    val_ds = ForecastDataset(
        bundle.val_X, bundle.val_label, cfg.seq_len, cfg.pred_len,
        cfg.horizon)
    test_ds = ForecastDataset(
        bundle.test_X, bundle.test_label, cfg.seq_len, cfg.pred_len,
        cfg.horizon)

    # 输入窗口 + horizon + 预测窗口覆盖的整段时间都必须正常
    span = cfg.seq_len + cfg.horizon + cfg.pred_len - 1
    normal = _normal_indices(bundle.train_label, span)
    train_ds = Subset(train_full, normal)

    if cfg.smoke:
        train_ds = Subset(train_ds, list(range(min(20000, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(4000, len(val_ds)))))
        test_ds = Subset(test_ds, list(range(min(20000, len(test_ds)))))

    return (_make_loader(train_ds, cfg, shuffle=True),
            _make_loader(val_ds, cfg, shuffle=False),
            _make_loader(test_ds, cfg, shuffle=False))


def make_reconstruction_loaders(bundle: DataBundle, cfg: RunConfig):
    """返回 train/val/test DataLoader；训练只使用正常窗口。"""
    train_full = ReconstructionDataset(
        bundle.train_X, bundle.train_label, cfg.seq_len)
    val_ds = ReconstructionDataset(
        bundle.val_X, bundle.val_label, cfg.seq_len)
    test_ds = ReconstructionDataset(
        bundle.test_X, bundle.test_label, cfg.seq_len)

    normal = _normal_indices(bundle.train_label, cfg.seq_len)
    train_ds = Subset(train_full, normal)

    if cfg.smoke:
        train_ds = Subset(train_ds, list(range(min(20000, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(4000, len(val_ds)))))
        test_ds = Subset(test_ds, list(range(min(20000, len(test_ds)))))

    return (_make_loader(train_ds, cfg, shuffle=True),
            _make_loader(val_ds, cfg, shuffle=False),
            _make_loader(test_ds, cfg, shuffle=False))

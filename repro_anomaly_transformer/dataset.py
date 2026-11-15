# -*- coding: utf-8 -*-
"""SWaT/WADI 的滑窗 Dataset，对应原仓库的 *SegLoader。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset):
    """输入 X [N,C]，返回窗口 [win,C] 和窗口内逐点标签 [win]。

    mode:
      - train / test / val: 起点间隔 step（官方 SWaT 的 step=1）
      - thre: 非重叠窗口，起点间隔 win_size（官方 test 评估模式）
    """

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None,
                 win_size: int = 100, step: int = 1, mode: str = "train",
                 clip_value: float | None = None):
        # 保留 memmap，避免把整个 train_X 一次性读进匿名内存；
        # 每次 __getitem__ 只 copy 一个 100×C 的小窗口。
        self.X = X
        self.y = None if y is None else np.asarray(y).reshape(-1)
        self.win_size = int(win_size)
        self.step = int(step)
        self.mode = mode
        self.clip_value = (None if clip_value is None or clip_value <= 0
                           else float(clip_value))
        if len(self.X) < self.win_size:
            raise ValueError(f"数据长度 {len(self.X)} < win_size {self.win_size}")

    def __len__(self) -> int:
        if self.mode == "thre":
            return (len(self.X) - self.win_size) // self.win_size + 1
        return (len(self.X) - self.win_size) // self.step + 1

    def __getitem__(self, index: int):
        if self.mode == "thre":
            start = index * self.win_size
        else:
            start = index * self.step
        x = np.array(self.X[start:start + self.win_size],
                     dtype=np.float32, copy=True)
        if self.clip_value is not None:
            np.clip(x, -self.clip_value, self.clip_value, out=x)
        if self.y is None:
            y = np.zeros(self.win_size, dtype=np.float32)
        else:
            y = np.array(self.y[start:start + self.win_size],
                         dtype=np.float32, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_prepared(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    return {
        # 大数组用只读 memmap，降低常驻内存
        "train_X": np.load(data_dir / "train_X.npy", mmap_mode="r"),
        "test_X": np.load(data_dir / "test_X.npy", mmap_mode="r"),
        "test_y": np.load(data_dir / "test_y.npy"),
        "test_timestamp": np.load(data_dir / "test_timestamp.npy"),
        "train_timestamp": np.load(data_dir / "train_timestamp.npy"),
    }


def make_loader(X, y=None, win_size=100, step=1, mode="train",
                batch_size=32, shuffle=None,
                clip_value=None) -> DataLoader:
    ds = WindowDataset(X, y, win_size=win_size, step=step, mode=mode,
                       clip_value=clip_value)
    if shuffle is None:
        shuffle = mode == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, drop_last=False)

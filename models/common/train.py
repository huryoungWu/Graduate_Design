# -*- coding: utf-8 -*-
"""共享训练循环。

预测类：默认 MSE(pred, y)；模型若实现 ``training_loss(x, y)`` 则优先使用
        （例如 Seq2Seq 的 teacher forcing）。
重构类：调用模型的 training_loss(x)，具体损失由模型自己定义。
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
from tqdm import tqdm


def train_forecaster(model, loader, epochs: int, lr: float,
                     device: torch.device, name: str) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs))
    criterion = nn.MSELoss()
    # 模型可选实现 training_loss(x, y)（例如 Seq2Seq 的 teacher forcing）
    custom_loss = getattr(model, "training_loss", None)
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total, n_seen = 0.0, 0
        pbar = tqdm(loader, desc=f"[{name}] Epoch {epoch:02d}/{epochs}",
                    unit="batch", ncols=110, dynamic_ncols=True)
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if callable(custom_loss):
                loss = custom_loss(x, y)
            else:
                loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "avg": f"{total / max(n_seen, 1):.4f}"})
        pbar.close()
        scheduler.step()
        print(f"[{name}] Epoch {epoch:02d} | Train MSE: "
              f"{total / max(n_seen, 1):.6f}")
    return time.time() - start


def train_reconstructor(model, loader, epochs: int, lr: float,
                        device: torch.device, name: str) -> float:
    # 模型可提供自定义训练循环（例如 USAD 需要两个优化器和
    # epoch 相关的对抗损失权重），签名:
    #   fit(loader, epochs, lr, device, name) -> 训练耗时（秒）
    custom_fit = getattr(model, "fit", None)
    if callable(custom_fit):
        print(f"[{name}] 使用模型自定义训练循环 fit()")
        return float(custom_fit(loader, epochs, lr, device, name))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs))
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total, n_seen = 0.0, 0
        pbar = tqdm(loader, desc=f"[{name}] Epoch {epoch:02d}/{epochs}",
                    unit="batch", ncols=110, dynamic_ncols=True)
        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "avg": f"{total / max(n_seen, 1):.4f}"})
        pbar.close()
        scheduler.step()
        print(f"[{name}] Epoch {epoch:02d} | Train loss: "
              f"{total / max(n_seen, 1):.6f}")
    return time.time() - start

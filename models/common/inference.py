# -*- coding: utf-8 -*-
"""统一的模型推理与输出收集。

这里只负责“把模型跑一遍并收集数组”，不做阈值、不做评估。
Stage 1 的 runner 调用这些函数，然后把结果交给 artifacts 保存。
"""
from __future__ import annotations

import numpy as np
import torch

from .eval import apply_fusion


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _concat(values: list[np.ndarray]) -> np.ndarray | None:
    if not values:
        return None
    return np.concatenate(values, axis=0)


@torch.no_grad()
def collect_forecast_outputs(model, loader, device,
                             first_only: bool = True):
    """收集预测类的 pred/true/label。

    Parameters
    ----------
    first_only:
        True 时只返回第一步 ``[:, 0, :]``，形状 [N, C]；
        False 时返回全部预测步，形状 [N, P, C]。

    Returns
    -------
    pred, true, label
    """
    preds, trues, labs = [], [], []
    model.eval()
    for x, y, lab in loader:
        out = model(x.to(device, non_blocking=True))
        if first_only:
            preds.append(_to_numpy(out[:, 0, :]))
            trues.append(_to_numpy(y[:, 0, :]))
            labs.append(_to_numpy(lab[:, 0] if lab.ndim > 1 else lab))
        else:
            preds.append(_to_numpy(out))
            trues.append(_to_numpy(y))
            labs.append(_to_numpy(lab))
    return (_concat(preds), _concat(trues), _concat(labs))


def _reconstruction_from_model(model, x: torch.Tensor):
    """尽力从模型 forward 取重构张量；取不到返回 None。

    - AE: forward 返回 Tensor [B,L,C]
    - VAE: forward 返回 (recon, mu, logvar)
    - AnomalyTransformer: forward 返回 (recon, series, prior)
    - DCdetector/SBT: 返回表示或 (z1,z2)，形状不等于输入，跳过
    """
    try:
        out = model(x)
    except Exception:
        return None
    if isinstance(out, tuple):
        out = out[0]
    if torch.is_tensor(out) and out.shape == x.shape:
        return out
    return None


@torch.no_grad()
def collect_recon_outputs(model, loader, device):
    """收集重构/对比类的分数与（可用的）重构数组。

    Returns
    -------
    dict with keys:
        window_scores: {name: [N]}
        point_scores:  {name: [N]}
        ts_scores:     {name: [N,L]}
        channel:       {name: [N,C]}   # abs_last / sq_last / recon_last / true_last
        label_window:  [N]
    """
    window: dict[str, list[np.ndarray]] = {}
    point: dict[str, list[np.ndarray]] = {}
    ts: dict[str, list[np.ndarray]] = {}
    channel: dict[str, list[np.ndarray]] = {}
    labels: list[np.ndarray] = []

    model.eval()
    for x, lab in loader:
        x = x.to(device, non_blocking=True)
        out = model.score_batch(x)

        for name, value in out.get("window", {}).items():
            window.setdefault(name, []).append(_to_numpy(value))
        for name, value in out.get("point", {}).items():
            point.setdefault(name, []).append(_to_numpy(value))
        for name, value in out.get("ts", {}).items():
            ts.setdefault(name, []).append(_to_numpy(value))
        for name, value in out.get("channel", {}).items():
            channel.setdefault(name, []).append(_to_numpy(value))

        # 模型没有显式提供 ts 时，尝试从 forward 取重构补上
        if not out.get("ts"):
            recon = _reconstruction_from_model(model, x)
            if recon is not None:
                diff = recon - x
                ts.setdefault("abs_ts", []).append(
                    _to_numpy(diff.abs().mean(dim=-1)))
                ts.setdefault("sq_ts", []).append(
                    _to_numpy((diff ** 2).mean(dim=-1)))
                if "recon_last" not in channel:
                    channel.setdefault("recon_last", []).append(
                        _to_numpy(recon[:, -1, :]))
                    channel.setdefault("true_last", []).append(
                        _to_numpy(x[:, -1, :]))

        labels.append(_to_numpy(lab))

    window_scores = {k: _concat(v) for k, v in window.items()}
    point_scores = {k: _concat(v) for k, v in point.items()}
    ts_scores = {k: _concat(v) for k, v in ts.items()}
    channel_arrays = {k: _concat(v) for k, v in channel.items()}

    # AE/VAE 已有 abs_last/sq_last；AnomalyTransformer 只给 recon_last/true_last
    if ("recon_last" in channel_arrays
            and "true_last" in channel_arrays):
        diff = channel_arrays["recon_last"] - channel_arrays["true_last"]
        if "abs_last" not in channel_arrays:
            channel_arrays["abs_last"] = np.abs(diff)
        if "sq_last" not in channel_arrays:
            channel_arrays["sq_last"] = diff ** 2

    return {
        "window_scores": window_scores,
        "point_scores": point_scores,
        "ts_scores": ts_scores,
        "channel": channel_arrays,
        "label_window": _concat(labels),
    }


def fuse_val_test(model, val_scores: dict, test_scores: dict
                  ) -> tuple[dict, dict]:
    """调用模型的 make_fused（如果存在），返回融合后的分数字典。"""
    return apply_fusion(model, val_scores, test_scores)

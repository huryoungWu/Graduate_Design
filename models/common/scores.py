# -*- coding: utf-8 -*-
"""残差计算与异常分数构造。

Stage 2 只依赖这里的纯数组函数，不依赖模型类。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .protocol import rolling_adaptive_normalize


def aggregate_channels(residual: np.ndarray, how: str) -> np.ndarray:
    """对 [N, C] 的逐通道残差做聚合。"""
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 2:
        raise ValueError(f"期望 [N,C]，实际 {residual.shape}")
    if how == "mean":
        return residual.mean(axis=1)
    if how == "max":
        return residual.max(axis=1)
    if how == "top3_mean":
        k = min(3, residual.shape[1])
        return np.sort(residual, axis=1)[:, -k:].mean(axis=1)
    raise ValueError(f"未知通道聚合方式: {how}")


def forecast_candidates(pred: np.ndarray, true: np.ndarray,
                        channel_aggs=("mean", "max", "top3_mean"),
                        ref_pred: np.ndarray | None = None,
                        ref_true: np.ndarray | None = None,
                        ref_labels: np.ndarray | None = None
                        ) -> dict[str, np.ndarray]:
    """预测类：|pred - true| -> 通道聚合。

    pred/true 形状 [N, C]（第一步预测）。

    如果提供 ``ref_pred/ref_true/ref_labels``，额外计算逐通道归一化的
    ``channel_norm_*`` 分数：先用参考集（验证集）正常点的逐通道 99% 分位
    做尺度归一化，再取 max/top3/mean。这样可缓解 mean 被多通道稀释的问题。
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    residual = np.abs(pred - true)
    out = {how: aggregate_channels(residual, how) for how in channel_aggs}

    if ref_pred is not None and ref_true is not None:
        ref_resid = np.abs(np.asarray(ref_pred, dtype=np.float64)
                           - np.asarray(ref_true, dtype=np.float64))
        if ref_labels is None:
            ref_labels = np.zeros(len(ref_resid), dtype=np.int64)
        normal = ref_resid[np.asarray(ref_labels).astype(int) == 0]
        if len(normal):
            scale = np.quantile(normal, 0.99, axis=0)
            scale = np.maximum(scale, 1e-6)
            norm_resid = residual / scale
            out["channel_norm_max"] = norm_resid.max(axis=1)
            k = min(3, norm_resid.shape[1])
            out["channel_norm_top3"] = np.sort(
                norm_resid, axis=1)[:, -k:].mean(axis=1)
            out["channel_norm_mean"] = norm_resid.mean(axis=1)
    return out


def recon_window_candidates(artifact: dict,
                            aggs=("mean", "max", "top3_mean")
                            ) -> dict[str, np.ndarray]:
    """重构/对比类的窗口级候选分数。

    优先使用 Stage 1 保存的模型特有分数（recon_max、score_max、fused 等）；
    如果存在逐时刻误差 ``sq_ts``/``abs_ts``，再补充 mean/max/top-k 聚合。
    """
    out = dict(artifact.get("window_scores", {}))
    ts = artifact.get("ts_scores", {})

    sq_ts = ts.get("sq_ts")
    if sq_ts is not None:
        sq_ts = np.asarray(sq_ts, dtype=np.float64)
        out.setdefault("recon_max", sq_ts.max(axis=1))
        out.setdefault("recon_mean", sq_ts.mean(axis=1))
        if "top3_mean" in aggs:
            k = min(3, sq_ts.shape[1])
            out.setdefault("recon_top3_mean",
                           np.sort(sq_ts, axis=1)[:, -k:].mean(axis=1))

    abs_ts = ts.get("abs_ts")
    if abs_ts is not None:
        abs_ts = np.asarray(abs_ts, dtype=np.float64)
        out.setdefault("abs_max", abs_ts.max(axis=1))
        out.setdefault("abs_mean", abs_ts.mean(axis=1))

    return out


def recon_point_candidates(artifact: dict,
                           ref_artifact: dict | None = None
                           ) -> dict[str, np.ndarray]:
    """重构类的逐点候选分数（窗口最后一点）。

    提供 ``ref_artifact`` 时，用参考集正常点的逐通道 99% 分位做尺度归一化，
    额外生成 ``channel_norm_*`` 分数。
    """
    out = dict(artifact.get("point_scores", {}))
    channel = artifact.get("channel", {})

    # artifact 只保存 recon_last/true_last，这里按需计算 abs/sq
    if ("recon_last" in channel and "true_last" in channel
            and ("abs_last" not in channel or "sq_last" not in channel)):
        diff = (np.asarray(channel["recon_last"], dtype=np.float64)
                - np.asarray(channel["true_last"], dtype=np.float64))
        channel = dict(channel)
        channel.setdefault("abs_last", np.abs(diff))
        channel.setdefault("sq_last", diff ** 2)

    abs_last = channel.get("abs_last")
    if abs_last is not None:
        abs_last = np.asarray(abs_last, dtype=np.float64)
        out.setdefault("point_recon_mae_last", abs_last.mean(axis=1))
    sq_last = channel.get("sq_last")
    if sq_last is not None:
        out.setdefault("point_recon_mse_last",
                       np.asarray(sq_last, dtype=np.float64).mean(axis=1))

    # 逐通道归一化分数
    if abs_last is not None and ref_artifact is not None:
        ref_channel = ref_artifact.get("channel", {})
        ref_abs = ref_channel.get("abs_last")
        if (ref_abs is None and "recon_last" in ref_channel
                and "true_last" in ref_channel):
            ref_diff = (np.asarray(ref_channel["recon_last"], dtype=np.float64)
                        - np.asarray(ref_channel["true_last"],
                                     dtype=np.float64))
            ref_abs = np.abs(ref_diff)
        ref_labels = ref_artifact.get("label_point")
        if ref_abs is not None and ref_labels is not None:
            ref_abs = np.asarray(ref_abs, dtype=np.float64)
            normal = ref_abs[np.asarray(ref_labels).astype(int) == 0]
            if len(normal):
                scale = np.maximum(np.quantile(normal, 0.99, axis=0), 1e-6)
                norm = abs_last / scale
                out["channel_norm_max"] = norm.max(axis=1)
                k = min(3, norm.shape[1])
                out["channel_norm_top3"] = np.sort(
                    norm, axis=1)[:, -k:].mean(axis=1)
                out["channel_norm_mean"] = norm.mean(axis=1)
    return out


def smooth_score(score: np.ndarray, window: int) -> np.ndarray:
    """沿时间做滑动平均；window<=1 时原样返回。"""
    if window is None or int(window) <= 1:
        return np.asarray(score, dtype=np.float64)
    s = pd.Series(np.asarray(score, dtype=np.float64))
    return s.rolling(int(window), min_periods=1).mean().to_numpy()


def normalize_pair(val_score: np.ndarray, test_score: np.ndarray,
                   method: str = "none",
                   window: int = 3600) -> tuple[np.ndarray, np.ndarray]:
    """对 val/test 分数做可选归一化。

    目前支持：
      - none：不处理
      - rolling：因果滚动 median/MAD 标准化（每个 split 独立）
    """
    if method in (None, "", "none"):
        return (np.asarray(val_score, dtype=np.float64),
                np.asarray(test_score, dtype=np.float64))
    if method != "rolling":
        raise ValueError(f"未知归一化方式: {method}")
    out = rolling_adaptive_normalize(
        {"val": val_score, "test": test_score},
        window=window, min_periods=min(120, max(2, window // 30)))
    return out["val"], out["test"]

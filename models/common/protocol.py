# -*- coding: utf-8 -*-
"""SWaT 异常检测统一评估协议。

核心原则
--------
1. 训练/校准：只使用原始 Normal 段（``period == "normal"``）；
2. 验证集：Attack 段前 ``VAL_FRAC``（默认 20%），用于选择阈值和评分配置；
3. 测试集：Attack 段剩余部分，只用于最终报告；
4. 测试集标签不得参与阈值、评分配置、模型或变体的任何选择。

SWaT 的 ``SWaT_all_1s.csv`` 中前 495,000 行来自原始 Normal 文件，
后 449,919 行来自原始 Attack 文件；按 ``period`` 列切分即可对齐原始基准。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "VAL_FRAC",
    "split_train_val_test",
    "select_threshold",
    "select_threshold_fpr",
    "evaluate_fixed",
    "select_best",
    "aggregate_residual",
    "rolling_adaptive_normalize",
    "print_protocol",
]

VAL_FRAC = 0.20


def split_train_val_test(df: pd.DataFrame, val_frac: float = VAL_FRAC):
    """按 period 边界切分：Normal 段训练，Attack 段验证 + 测试。

    返回 ``(train_df, val_df, test_df, split_idx, val_end)``。
    """
    if "period" in df.columns:
        attack_idx = df.index[df["period"].astype(str).str.lower() == "attack"]
        split_idx = int(attack_idx[0]) if len(attack_idx) else int(len(df) * 0.7)
    else:
        pos = np.flatnonzero(df["label"].astype(int).to_numpy() == 1)
        split_idx = int(pos[0]) if len(pos) else int(len(df) * 0.7)

    val_end = split_idx + int((len(df) - split_idx) * float(val_frac))
    return (
        df.iloc[:split_idx].reset_index(drop=True),
        df.iloc[split_idx:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
        split_idx,
        val_end,
    )


def select_threshold(score, labels):
    """在验证集上按 F1 选阈值。测试集标签不得传入此函数。"""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    if len(np.unique(labels)) < 2:
        return float(np.quantile(score, 0.99)), float("nan")
    precisions, recalls, thresholds = precision_recall_curve(labels, score)
    if len(thresholds) == 0:
        return float(np.quantile(score, 0.99)), float("nan")
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    k = min(int(np.argmax(f1s)), len(thresholds) - 1)
    return float(thresholds[k]), float(f1s[k])


def select_threshold_fpr(score, labels, target_fpr=0.01):
    """固定 FPR 阈值：取验证集**正常点**分数的 ``1-target_fpr`` 分位。

    相比 max-F1 阈值，这个口径对类别比例和阈值抖动更稳健；只使用
    验证集正常点的分数分布，不使用攻击点标签。
    """
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    normal = score[labels == 0]
    if normal.size == 0:
        return float(np.quantile(score, 1.0 - float(target_fpr))), float("nan")
    return float(np.quantile(normal, 1.0 - float(target_fpr))), float("nan")


def evaluate_fixed(name, score, labels, threshold, threshold_source="val_best_f1"):
    """测试集评估：阈值由外部给定，这里不再用测试标签调阈值。"""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    alarm = (score >= float(threshold)).astype(np.int64)
    normal = labels == 0
    return {
        "name": name,
        "AUC": float(roc_auc_score(labels, score)),
        "PR_AUC": float(average_precision_score(labels, score)),
        "Precision": float(precision_score(labels, alarm, zero_division=0)),
        "Recall": float(recall_score(labels, alarm, zero_division=0)),
        "F1": float(f1_score(labels, alarm, zero_division=0)),
        "FPR": float(alarm[normal].mean()) if normal.any() else float("nan"),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
    }


def select_best(scores: dict, labels):
    """在验证集上为每个评分配置选阈值，并选出验证集 F1 最高的配置。"""
    thresholds, val_f1 = {}, {}
    for name, score in scores.items():
        thresholds[name], val_f1[name] = select_threshold(score, labels)
    finite = {k: v for k, v in val_f1.items() if np.isfinite(v)}
    best_name = max(finite, key=finite.get) if finite else next(iter(val_f1))
    return best_name, thresholds, val_f1


def aggregate_residual(residual):
    """对 ``[N, C]`` 残差做逐点聚合：mean / max / top3_mean。"""
    residual = np.asarray(residual, dtype=np.float64)
    top3 = np.sort(residual, axis=1)[:, -3:]
    return {
        "mean": residual.mean(axis=1),
        "max": residual.max(axis=1),
        "top3_mean": top3.mean(axis=1),
    }


def rolling_adaptive_normalize(scores: dict, window: int = 3600,
                               min_periods: int = 120) -> dict:
    """因果滚动鲁棒标准化，用于抵消测试期内的分数漂移。

    对每个分数序列 ``s_t`` 计算::

        s'_t = (s_t - median(s_{t-W..t-1})) / (1.4826 * MAD(s_{t-W..t-1}))

    只使用当前时刻之前的数据（``shift(1)``），不会引入未来信息；中位数/MAD
    对短暂攻击不敏感。窗口按**样本点**计；如果输入是抽样的校准分数，
    调用方需要把窗口按抽样步长折算。
    """
    if window is None or window <= 0:
        return {k: np.asarray(v, dtype=np.float64) for k, v in scores.items()}

    window = int(window)
    out = {}
    for name, arr in scores.items():
        s = pd.Series(np.asarray(arr, dtype=np.float64))
        past = s.shift(1)
        med = past.rolling(window, min_periods=min_periods).median()
        med = med.fillna(past.expanding(min_periods=1).median())
        dev = (past - med).abs()
        mad = dev.rolling(window, min_periods=min_periods).median()
        scale = 1.4826 * mad

        vals = s.to_numpy()
        global_mad = 1.4826 * float(np.nanmedian(np.abs(vals - np.nanmedian(vals))))
        if not np.isfinite(global_mad) or global_mad <= 1e-8:
            global_mad = float(np.nanstd(vals))
        if not np.isfinite(global_mad) or global_mad <= 1e-8:
            global_mad = 1.0
        # 局部 MAD 过小（安静期）会把微小波动放大成极端值：
        # 用全局 MAD 的 5% 作为尺度下限，并给热身期回退到全局尺度。
        floor = max(1e-6, 0.05 * global_mad)
        scale = scale.fillna(global_mad).clip(lower=floor)

        out[name] = ((s - med) / scale).fillna(0.0).to_numpy(dtype=np.float64)
    return out


def print_protocol(train_df, val_df, test_df, extra: str = ""):
    """打印协议摘要，便于在日志中留痕。"""
    print(
        "[protocol] train=[0,{},) | val={} ~ {} | test={} ~ {}{}".format(
            len(train_df),
            val_df["timestamp"].iloc[0], val_df["timestamp"].iloc[-1],
            test_df["timestamp"].iloc[0], test_df["timestamp"].iloc[-1],
            f" | {extra}" if extra else "",
        )
    )

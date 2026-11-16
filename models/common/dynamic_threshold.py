# -*- coding: utf-8 -*-
"""因果动态阈值：只使用当前时刻之前的历史分数，不使用标签/未来。

策略：
- static: Normal 段 q99（基线）
- rq_q_w: 过去 w 个窗口分数的 q 分位
- rz_k_w: 过去 w 个窗口 mean + k*std
- ewma_z_k_s: EWMA mean + k*EWMA std

所有滚动统计都 shift(1)，保证 t 时刻阈值只看到 t-1 及之前。
热身阶段（不足窗口）使用 Normal99 静态阈值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)

from models.reconstruction.usad_extended_metrics import point_adjust


def static_threshold(normal_scores, q=0.99) -> float:
    return float(np.percentile(np.asarray(normal_scores), q * 100.0))


def _fill(thr: pd.Series, warmup: float) -> np.ndarray:
    return thr.fillna(warmup).to_numpy(dtype=np.float64)


def rolling_quantile(scores, q, window, warmup) -> np.ndarray:
    s = pd.Series(np.asarray(scores, dtype=np.float64))
    thr = s.rolling(window, min_periods=max(10, window // 2)).quantile(
        q).shift(1)
    return _fill(thr, warmup)


def rolling_z(scores, k, window, warmup) -> np.ndarray:
    s = pd.Series(np.asarray(scores, dtype=np.float64))
    mean = s.rolling(window, min_periods=max(10, window // 2)).mean().shift(1)
    std = s.rolling(window, min_periods=max(10, window // 2)).std().shift(1)
    thr = mean + k * std.clip(lower=1e-9)
    return _fill(thr, warmup)


def ewma_z(scores, k, span, warmup) -> np.ndarray:
    s = pd.Series(np.asarray(scores, dtype=np.float64))
    mean = s.ewm(span=span, min_periods=max(10, span // 10)).mean().shift(1)
    std = s.ewm(span=span, min_periods=max(10, span // 10)).std().shift(1)
    thr = mean + k * std.clip(lower=1e-9)
    return _fill(thr, warmup)


def make_thresholds(scores, warmup) -> dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    out = {"static_N99": np.full(n, warmup, dtype=np.float64)}
    for q, w in ((0.99, 500), (0.99, 2000), (0.995, 1000)):
        out[f"rq{q:.3f}_w{w}"] = rolling_quantile(
            scores, q, w, warmup)
    for k, w in ((3.0, 1000), (4.0, 2000)):
        out[f"rz{k:.1f}_w{w}"] = rolling_z(scores, k, w, warmup)
    out["ewma_z3_s1000"] = ewma_z(scores, 3.0, 1000, warmup)
    return out


def metrics_dynamic(y, scores, thresholds) -> dict:
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    pred = (scores > thresholds).astype(np.int64)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    neg = y == 0
    fpr = float(((pred == 1) & neg).sum() / max(neg.sum(), 1))
    auc = (float(roc_auc_score(y, scores))
           if len(np.unique(y)) > 1 else np.nan)
    ap = (float(average_precision_score(y, scores))
          if len(np.unique(y)) > 1 else np.nan)
    pa = point_adjust(y, pred)
    pp, pr, pf1, _ = precision_recall_fscore_support(
        y, pa, average="binary", zero_division=0)
    return {
        "Precision": float(p), "Recall": float(r), "F1": float(f1),
        "FPR": fpr, "AUC": auc, "PR_AUC": ap,
        "PA_Precision": float(pp), "PA_Recall": float(pr),
        "PA_F1": float(pf1),
    }

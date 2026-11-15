# -*- coding: utf-8 -*-
"""对 USAD 论文协议已保存的窗口分数做补充指标计算（无需重新训练）。

读取 ``runs/usad_paper_*/window_scores.csv`` 与原始 CSV 的 Attack 段标签，
计算：

- 窗口级：AUC、ROC 等错误率阈值下的 P/R/F1、oracle 最优 F1；
- 逐点级：把窗口分数对齐到窗口最后一点，计算
  raw F1（阈值=EER / oracle）与 point-adjusted F1
  （阈值=EER / 窗口最优 / oracle 扫描）。

PA-F1 是文献中 USAD 常用的报告口径（SWaT ≈0.85、WADI ≈0.43 一类
数字通常来自该口径）；窗口级 AUC/F1 与论文表格不可直接比较。

用法::

    python -m models.reconstruction.usad_extended_metrics \
        --run-dir runs/usad_paper_swat \
        --csv 08_SWaT/SWaT_all_1s.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)


def eer_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    diff = tpr - (1.0 - fpr)
    idx = np.flatnonzero(np.diff(np.sign(diff)))
    if len(idx) == 0:
        idx = np.array([int(np.argmin(np.abs(diff)))])
    return float(thresholds[int(idx[0])])


def best_f1(labels: np.ndarray, scores: np.ndarray):
    prec, rec, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    k = int(np.nanargmax(f1[:-1]))
    return float(thresholds[k]), float(f1[k])


def prf(labels, pred):
    p, r, f1, _ = precision_recall_fscore_support(
        np.asarray(labels).astype(int), np.asarray(pred).astype(int),
        average="binary", zero_division=0)
    return float(p), float(r), float(f1)


def point_adjust(labels: np.ndarray, pred: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    out = np.asarray(pred).astype(int).copy()
    edges = np.flatnonzero(np.diff(np.r_[0, labels, 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if out[s:e].any():
            out[s:e] = 1
    return out


def sweep_best_pa_f1(labels: np.ndarray, scores: np.ndarray,
                     n_grid: int = 300):
    """在分数分位点上扫描，返回 PA-F1 最大的阈值。"""
    qs = np.linspace(0.0, 100.0, n_grid)
    candidates = np.unique(np.quantile(scores, qs / 100.0))
    best = (-1.0, float("nan"), 0.0, 0.0)
    for thr in candidates:
        pred = scores >= thr
        adj = point_adjust(labels, pred)
        p, r, f1 = prf(labels, adj)
        if f1 > best[0]:
            best = (f1, float(thr), p, r)
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="USAD 已保存分数的补充指标")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--out", default="",
                   help="结果 JSON；为空时写 <run-dir>/paper_metrics_extended.json")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    df = pd.read_csv(args.csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    attack = df[df["period"].astype(str).str.lower() == "attack"].copy()
    L = int(args.seq_len)

    ws = pd.read_csv(run_dir / "window_scores.csv")
    ws["timestamp"] = pd.to_datetime(ws["timestamp"])
    # 兼容 --smoke 截断后的 attack 段：只保留分数覆盖到的时间范围
    attack = attack[attack["timestamp"] <= ws["timestamp"].iloc[-1]]
    attack = attack.reset_index(drop=True)
    point_labels = attack["label"].to_numpy(np.int8)
    scores = ws["score"].to_numpy(np.float64)
    win_labels = ws["label_window"].to_numpy(np.int8)
    if len(scores) != len(attack) - L + 1:
        print(f"[warn] 窗口数 {len(scores)} != Attack 行数-L+1 "
              f"{len(attack) - L + 1}；按较短长度截断计算")
        n = min(len(scores), len(attack) - L + 1)
        scores = scores[:n]
        win_labels = win_labels[:n]
        point_labels = point_labels[:n + L - 1]

    # 窗口级
    win_auc = float(roc_auc_score(win_labels, scores))
    win_ap = float(average_precision_score(win_labels, scores))
    thr_eer = eer_threshold(win_labels, scores)
    p_eer, r_eer, f1_eer = prf(win_labels, scores >= thr_eer)
    thr_best, f1_best = best_f1(win_labels, scores)
    p_best, r_best, _ = prf(win_labels, scores >= thr_best)

    # 逐点（窗口分数对齐到窗口最后一点）
    valid = np.arange(len(point_labels)) >= (L - 1)
    pt_scores = np.full(len(point_labels), np.nan)
    pt_scores[L - 1:] = scores[:len(pt_scores) - L + 1]
    y = point_labels[valid]
    s = pt_scores[valid]

    p_raw, r_raw, f1_raw = prf(y, s >= thr_eer)
    p_pa, r_pa, f1_pa = prf(y, point_adjust(y, s >= thr_eer))
    p_pa_bestwin, r_pa_bestwin, f1_pa_bestwin = prf(
        y, point_adjust(y, s >= thr_best))
    thr_pt_best, f1_pt_best = best_f1(y, s)
    p_pt_best, r_pt_best, _ = prf(y, s >= thr_pt_best)
    f1_pa_oracle, thr_pa_oracle, p_pa_oracle, r_pa_oracle = sweep_best_pa_f1(
        y, s)

    result = {
        "run_dir": str(run_dir),
        "n_window": int(len(scores)),
        "n_point_valid": int(valid.sum()),
        "window_level": {
            "AUC": win_auc, "PR_AUC": win_ap,
            "threshold_eer": thr_eer,
            "Precision_eer": p_eer, "Recall_eer": r_eer, "F1_eer": f1_eer,
            "threshold_best_f1_oracle": thr_best,
            "Precision_best_f1_oracle": p_best,
            "Recall_best_f1_oracle": r_best,
            "F1_best_f1_oracle": f1_best,
        },
        "point_level_raw": {
            "threshold_eer": thr_eer,
            "Precision": p_raw, "Recall": r_raw, "F1": f1_raw,
            "threshold_oracle": thr_pt_best,
            "Precision_oracle": p_pt_best,
            "Recall_oracle": r_pt_best,
            "F1_oracle": f1_pt_best,
        },
        "point_level_PA": {
            "threshold_eer": thr_eer,
            "Precision": p_pa, "Recall": r_pa, "F1": f1_pa,
            "threshold_best_window_f1": thr_best,
            "Precision_best_window_f1": p_pa_bestwin,
            "Recall_best_window_f1": r_pa_bestwin,
            "F1_best_window_f1": f1_pa_bestwin,
            "threshold_oracle": thr_pa_oracle,
            "Precision_oracle": p_pa_oracle,
            "Recall_oracle": r_pa_oracle,
            "F1_oracle": f1_pa_oracle,
        },
    }

    out = Path(args.out) if args.out else (
        run_dir / "paper_metrics_extended.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[extended] 窗口级: AUC={win_auc:.4f} | "
          f"EER F1={f1_eer:.4f} | oracle F1={f1_best:.4f}")
    print(f"[extended] 逐点 raw: EER F1={f1_raw:.4f} | "
          f"oracle F1={f1_pt_best:.4f}")
    print(f"[extended] 逐点 PA: EER F1={f1_pa:.4f} | "
          f"best-window F1={f1_pa_bestwin:.4f} | "
          f"oracle F1={f1_pa_oracle:.4f}")
    print(f"[extended] 已保存: {out}")


if __name__ == "__main__":
    main()

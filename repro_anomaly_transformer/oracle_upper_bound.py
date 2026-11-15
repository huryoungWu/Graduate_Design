# -*- coding: utf-8 -*-
"""测试集 oracle 阈值上限（诊断用，使用测试标签，不可部署）。

对每个模型的分数，在测试集上同时搜索：
  - 方向（score 或 -score）
  - 阈值
分别最大化标准 F1 和 PA-F1，报告对应的 P/R/F1/FPR/AUC/PR-AUC。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score,
                             precision_recall_curve,
                             precision_recall_fscore_support,
                             roc_auc_score)

from models.reconstruction.usad_extended_metrics import point_adjust

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "runs" / "recon_comparison" / "ORACLE_UPPER_BOUND.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "ORACLE_UPPER_BOUND.md"


def _prf(y, p):
    p0, r0, f0, _ = precision_recall_fscore_support(
        y, p, average="binary", zero_division=0)
    return float(p0), float(r0), float(f0)


def _fpr(y, p):
    neg = y == 0
    return float(((p == 1) & neg).sum() / max(neg.sum(), 1))


def _best_f1(y, s):
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    k = int(np.argmax(f1[:-1])) if len(thr) else 0
    t = float(thr[k]) if len(thr) else float(np.max(s))
    pred = (s > t).astype(int)
    p, r, f = _prf(y, pred)
    return t, p, r, f


def _map_point_pred(y_point, pred_win, offset):
    pp = np.zeros(len(y_point), np.int8)
    m = min(len(pred_win), len(y_point) - offset)
    pp[offset:offset + m] = pred_win[:m]
    return pp


def _best_pa(y_win, s, y_point, offset, n_grid=200):
    qs = np.linspace(0, 100, n_grid)
    cand = np.unique(np.quantile(s, qs / 100.0))
    best = (-1.0, float("nan"), 0.0, 0.0, 0.0)
    for t in cand:
        pred = (s > t).astype(int)
        pp = _map_point_pred(y_point, pred, offset)
        pa = point_adjust(y_point, pp)
        p, r, f = _prf(y_point, pa)
        if f > best[0]:
            best = (f, float(t), p, r, 0.0)
    f, t, p, r, _ = best
    pred = (s > t).astype(int)
    pp = _map_point_pred(y_point, pred, offset)
    pa = point_adjust(y_point, pp)
    tp = int(((pa == 1) & (y_point == 1)).sum())
    fp = int(((pa == 1) & (y_point == 0)).sum())
    fn = int(((pa == 0) & (y_point == 1)).sum())
    tn = int(((pa == 0) & (y_point == 0)).sum())
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return t, p, r, f, fpr, (tp, fp, tn, fn)


def _load_cases():
    cases = []
    specs = [
        ("IsolationForest", ROOT / "runs/iforest_swat/test.npz",
         "win_iforest_mean", 12, None),
        ("USAD", ROOT / "runs/usad_swat/test.npz",
         "win_usad_score", 12, None),
        ("LSTM-VAE", ROOT / "runs/lstmvae_swat/test.npz",
         "win_recon_mean", 12, None),
    ]
    for name, path, key, win, off in specs:
        with np.load(path, allow_pickle=False) as z:
            s = np.asarray(z[key], dtype=np.float64)
            y = np.asarray(z["label_window"]).astype(int)
            yp = np.asarray(z["label_point"]).astype(int)
        n = min(len(s), len(y), len(yp))
        cases.append(dict(model=name, level=f"window{win}", score=key,
                          s=s[:n], y=y[:n], yp=yp[:n], offset=0,
                          note="point labels aligned per window"))

    # AT 12-step
    te = np.load(ROOT / "repro_anomaly_transformer/runs/swat_20_80/test_energy.npy")
    ts = pd.read_csv(ROOT / "repro_anomaly_transformer/runs/swat_20_80/test_scores.csv")
    yp = ts["label"].to_numpy(np.int8)[:len(te)].astype(int)
    w = 12
    s = np.lib.stride_tricks.sliding_window_view(te, w).mean(axis=1)
    y = np.lib.stride_tricks.sliding_window_view(yp, w).max(axis=1)
    cases.append(dict(model="AnomalyTransformer(paper)",
                      level="window12", score="association_energy",
                      s=s, y=y, yp=yp, offset=w - 1,
                      note="official energy; 12-step"))

    # SBT 60-step
    sbt = ROOT / "repro_anomaly_transformer/runs/sbt_20_80/test_scores.csv"
    if sbt.exists():
        d = pd.read_csv(sbt)
        s = d["score"].to_numpy(np.float64)
        y = d["label_window"].to_numpy(int)
        data_y = np.load(ROOT / "repro_anomaly_transformer/data/swat/test_y.npy")
        n_val = int(len(data_y) * 0.2)
        yp = data_y[n_val:].astype(int)
        cases.append(dict(model="SBT", level="window60", score="score_max",
                          s=s, y=y, yp=yp, offset=59,
                          note="60-step window score"))
    return cases


def main():
    rows = []
    for c in _load_cases():
        s, y, yp = c["s"], c["y"], c["yp"]
        auc_raw = (float(roc_auc_score(y, s))
                   if len(np.unique(y)) > 1 else np.nan)
        ap_raw = (float(average_precision_score(y, s))
                  if len(np.unique(y)) > 1 else np.nan)

        best_f1 = None
        best_pa = None
        for sign in (+1, -1):
            z = sign * s
            t, p, r, f = _best_f1(y, z)
            if best_f1 is None or f > best_f1[0]:
                best_f1 = (f, sign, t, p, r)
            tpa, pp, pr, pf, pfpr, counts = _best_pa(
                y, z, yp, c["offset"])
            if best_pa is None or pf > best_pa[0]:
                best_pa = (pf, sign, tpa, pp, pr, pfpr, counts)

        f1, sign_f1, thr_f1, p1, r1 = best_f1
        pf1, sign_pa, thr_pa, p2, r2, pfpr, counts = best_pa
        # 该方向下的 AUC/PR
        z = sign_f1 * s
        auc = (float(roc_auc_score(y, z))
               if len(np.unique(y)) > 1 else np.nan)
        ap = (float(average_precision_score(y, z))
              if len(np.unique(y)) > 1 else np.nan)
        pred_f1 = (z > thr_f1).astype(int)
        fpr_f1 = _fpr(y, pred_f1)

        rows.append({
            "model": c["model"], "level": c["level"],
            "score": c["score"], "sign": sign_f1,
            "AUC": auc, "PR_AUC": ap,
            "AUC_raw": auc_raw, "PR_AUC_raw": ap_raw,
            "best_F1": f1, "F1_threshold": thr_f1,
            "F1_Precision": p1, "F1_Recall": r1, "F1_FPR": fpr_f1,
            "best_PA_F1": pf1, "PA_threshold": thr_pa,
            "PA_sign": sign_pa, "PA_Precision": p2,
            "PA_Recall": r2, "PA_FPR": pfpr,
            "PA_TP": counts[0], "PA_FP": counts[1],
            "PA_TN": counts[2], "PA_FN": counts[3],
            "n": len(y), "positives": int(y.sum()),
            "note": c["note"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    cols = ["model", "level", "sign", "AUC", "PR_AUC",
            "best_F1", "F1_Precision", "F1_Recall", "F1_FPR",
            "best_PA_F1", "PA_Precision", "PA_Recall", "PA_FPR"]
    lines = ["# 测试集 oracle 阈值上限（诊断用，不可部署）", "",
             "- 使用测试标签搜索方向和阈值，分别最大化标准 F1 和 PA-F1",
             "- 模型权重未重新训练",
             "- 这不是可部署结果，只用于回答“当前分数下最好能到什么程度”",
             ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, x in df.iterrows():
        vals = []
        for col in cols:
            v = x[col]
            if isinstance(v, (float, np.floating)):
                vals.append("nan" if np.isnan(v) else f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")


if __name__ == "__main__":
    main()

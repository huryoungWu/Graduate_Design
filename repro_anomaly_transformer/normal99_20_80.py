# -*- coding: utf-8 -*-
"""Normal-99 + 20/80 冻结测试对比（SWaT）。

- 训练：Normal only
- 测试：Attack 后 80%，冻结
- 阈值：Normal 分数的 99% 分位；测试集完全不参与
- 评估：12 步窗口（IF/USAD/LSTM-VAE/AT）或 60 步窗口（SBT）
- 同时报告标准 F1 和 PA-F1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)
from torch.utils.data import DataLoader

from models.reconstruction.sbt import build_model as build_sbt
from models.reconstruction.usad_extended_metrics import point_adjust

from .dataset import WindowDataset, load_prepared

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "runs" / "recon_comparison" / "COMPARISON_NORMAL99_20_80.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "COMPARISON_NORMAL99_20_80.md"


def _prf(y, p):
    p0, r0, f0, _ = precision_recall_fscore_support(
        y, p, average="binary", zero_division=0)
    return float(p0), float(r0), float(f0)


def _standard(y, s, thr):
    pred = (s > thr).astype(int)
    p, r, f1 = _prf(y, pred)
    neg = y == 0
    fpr = float(((pred == 1) & neg).sum() / max(neg.sum(), 1))
    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    ap = (float(average_precision_score(y, s))
          if len(np.unique(y)) > 1 else float("nan"))
    return pred, p, r, f1, fpr, auc, ap


def _pa_from_window_pred(y_point, pred_window, win_size, offset=None):
    """把窗口预测映射到窗口最后一点，再做 point-adjust。"""
    T = len(y_point)
    point_pred = np.zeros(T, dtype=np.int8)
    if offset is None:
        offset = win_size - 1
    m = min(len(pred_window), T - offset)
    point_pred[offset:offset + m] = pred_window[:m]
    pa = point_adjust(y_point, point_pred)
    return _prf(y_point, pa)


def main():
    rows = []
    # ---------------- IF / USAD / LSTM-VAE ----------------
    th = pd.read_csv(ROOT / "runs/normal99_full_attack/comparison.csv")
    th = th[(th.dataset == "SWaT") & (th.method == "q99_all")]
    th = th.set_index("model")["threshold"].to_dict()
    specs = [
        ("IsolationForest", ROOT / "runs/iforest_swat/test.npz",
         "win_iforest_mean", 12),
        ("USAD", ROOT / "runs/usad_swat/test.npz",
         "win_usad_score", 12),
        ("LSTM-VAE", ROOT / "runs/lstmvae_swat/test.npz",
         "win_recon_mean", 12),
    ]
    for name, path, key, win in specs:
        with np.load(path, allow_pickle=False) as z:
            score = np.asarray(z[key], dtype=np.float64)
            y_win = np.asarray(z["label_window"]).astype(int)
            y_point = np.asarray(z["label_point"]).astype(int)
        n = min(len(score), len(y_win))
        score, y_win = score[:n], y_win[:n]
        thr = float(th[name])
        pred, p, r, f1, fpr, auc, ap = _standard(y_win, score, thr)
        # 这里的 label_point 已经是“窗口最后一点”的逐秒标签，
        # 与窗口预测一一对齐；直接做 point-adjust。
        pa = point_adjust(y_point, pred)
        pa_p, pa_r, pa_f1 = _prf(y_point, pa)
        rows.append({
            "model": name, "level": f"window{win}",
            "score": key, "threshold": thr,
            "AUC": auc, "PR_AUC": ap,
            "Precision": p, "Recall": r, "F1": f1, "FPR": fpr,
            "PA_F1": pa_f1,
            "n": int(n), "positives": int(y_win.sum()),
            "note": "Normal q99 (full Normal); frozen 20/80 test",
        })

    # ---------------- Anomaly-Transformer（论文实现） ----------------
    tr = np.load(ROOT / "repro_anomaly_transformer/runs/swat/train_energy.npy")
    te = np.load(ROOT / "repro_anomaly_transformer/runs/swat_20_80/test_energy.npy")
    ts = pd.read_csv(ROOT / "repro_anomaly_transformer/runs/swat_20_80/test_scores.csv")
    y_point = ts["label"].to_numpy()[:len(te)].astype(int)
    w = 12
    tr_w = np.lib.stride_tricks.sliding_window_view(tr, w).mean(axis=1)
    te_w = np.lib.stride_tricks.sliding_window_view(te, w).mean(axis=1)
    y_w = np.lib.stride_tricks.sliding_window_view(y_point, w).max(axis=1)
    thr = float(np.percentile(tr_w, 99))
    pred, p, r, f1, fpr, auc, ap = _standard(y_w, te_w, thr)
    pa_p, pa_r, pa_f1 = _pa_from_window_pred(y_point, pred, w)
    rows.append({
        "model": "AnomalyTransformer(paper)", "level": f"window{w}",
        "score": "association_energy", "threshold": thr,
        "AUC": auc, "PR_AUC": ap,
        "Precision": p, "Recall": r, "F1": f1, "FPR": fpr,
        "PA_F1": pa_f1,
        "n": int(len(te_w)), "positives": int(y_w.sum()),
        "note": "Normal q99 of 12-step energy; frozen 20/80 test",
    })

    # ---------------- SBT ----------------
    data = load_prepared(ROOT / "repro_anomaly_transformer/data/swat")
    train_x = data["train_X"]
    test_x = data["test_X"]
    test_y = data["test_y"]
    n_val = int(len(test_x) * 0.2)
    test_x, test_y = test_x[n_val:], test_y[n_val:]
    sbt_dir = ROOT / "repro_anomaly_transformer/runs/sbt_20_80"
    ckpt = sbt_dir / "sbt_checkpoint.pth"
    if ckpt.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_sbt(train_x.shape[1], 60).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))
        model.eval()
        ds = WindowDataset(train_x, None, win_size=60, step=1, mode="train")
        loader = DataLoader(ds, batch_size=128, shuffle=False,
                            num_workers=0, pin_memory=True)
        normal_scores = []
        torch.manual_seed(42)
        with torch.no_grad():
            for x, _ in loader:
                out = model.score_batch(x.to(device, non_blocking=True))
                normal_scores.append(out["window"]["score_max"].cpu().numpy())
        normal_scores = np.concatenate(normal_scores).reshape(-1)
        sbt = pd.read_csv(sbt_dir / "test_scores.csv")
        score = sbt["score"].to_numpy(np.float64)
        y_win = sbt["label_window"].to_numpy(int)
        thr = float(np.percentile(normal_scores, 99))
        pred, p, r, f1, fpr, auc, ap = _standard(y_win, score, thr)
        pa_p, pa_r, pa_f1 = _pa_from_window_pred(test_y, pred, 60)
        rows.append({
            "model": "SBT", "level": "window60",
            "score": "score_max", "threshold": thr,
            "AUC": auc, "PR_AUC": ap,
            "Precision": p, "Recall": r, "F1": f1, "FPR": fpr,
            "PA_F1": pa_f1,
            "n": int(len(score)), "positives": int(y_win.sum()),
            "note": "Normal q99 of score_max; frozen 20/80 test",
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["level", "F1"], ascending=[True, False])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    cols = ["model", "level", "score", "threshold", "AUC", "PR_AUC",
            "Precision", "Recall", "F1", "FPR", "PA_F1", "n", "positives"]
    lines = ["# Normal-99 + 20/80 冻结测试对比（SWaT）", "",
             "- 训练：Normal only",
             "- 测试：Attack 后 80%，冻结",
             "- 阈值：Normal 分数的 99% 分位（测试集完全不参与）",
             "- IF/USAD/LSTM-VAE/AT 为 12 步窗口；SBT 为 60 步窗口",
             "- `F1` 是标准 F1；`PA_F1` 是 point-adjusted F1", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, x in df.iterrows():
        vals = []
        for c in cols:
            v = x[c]
            if isinstance(v, (float, np.floating)):
                vals.append("nan" if np.isnan(v) else f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")


if __name__ == "__main__":
    main()

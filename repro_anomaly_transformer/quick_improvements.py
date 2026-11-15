# -*- coding: utf-8 -*-
"""不重训的快速改进实验（SWaT，20/80 冻结测试）。

1. IF/USAD/LSTM-VAE/AT：用 Attack 前 20% 验证集选阈值（val_best_f1）
2. AT：改用纯重构误差作为分数
3. SBT：在验证集上比较 score 与 -score，选方向
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

from models.common.protocol import select_threshold
from models.reconstruction.sbt import build_model as build_sbt
from models.reconstruction.usad_extended_metrics import point_adjust

from .dataset import WindowDataset, load_prepared
from .vendor.AnomalyTransformer import AnomalyTransformer

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "runs" / "recon_comparison" / "IMPROVEMENTS_QUICK.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "IMPROVEMENTS_QUICK.md"


def _prf(y, p):
    p0, r0, f0, _ = precision_recall_fscore_support(
        y, p, average="binary", zero_division=0)
    return float(p0), float(r0), float(f0)


def _eval_window(y, s, thr, y_point=None, win=12, point_offset=None):
    pred = (s > thr).astype(int)
    p, r, f1 = _prf(y, pred)
    neg = y == 0
    fpr = float(((pred == 1) & neg).sum() / max(neg.sum(), 1))
    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan
    ap = (float(average_precision_score(y, s))
          if len(np.unique(y)) > 1 else np.nan)
    if y_point is None:
        pa_f1 = np.nan
    elif point_offset is None:
        # y_point 已经是“窗口最后一点”的逐窗口标签，与 pred 一一对齐
        pa = point_adjust(y_point[:len(pred)], pred)
        _, _, pa_f1 = _prf(y_point[:len(pred)], pa)
    else:
        point_pred = np.zeros(len(y_point), np.int8)
        off = point_offset
        m = min(len(pred), len(y_point) - off)
        point_pred[off:off + m] = pred[:m]
        pa = point_adjust(y_point, point_pred)
        _, _, pa_f1 = _prf(y_point, pa)
    return dict(Precision=p, Recall=r, F1=f1, FPR=fpr,
                AUC=auc, PR_AUC=ap, PA_F1=float(pa_f1)), pred


def _load_val_test(name, path, key):
    out = {}
    for split in ("val", "test"):
        with np.load(Path(path) / f"{split}.npz", allow_pickle=False) as z:
            out[split] = {
                "score": np.asarray(z[key], dtype=np.float64),
                "y_win": np.asarray(z["label_window"]).astype(int),
                "y_point": np.asarray(z["label_point"]).astype(int),
            }
    return out


def main():
    rows = []

    # ---------------- 1) 验证集阈值：IF / USAD / LSTM-VAE ----------------
    specs = [
        ("IsolationForest", ROOT / "runs/iforest_swat", "win_iforest_mean"),
        ("USAD", ROOT / "runs/usad_swat", "win_usad_score"),
        ("LSTM-VAE", ROOT / "runs/lstmvae_swat", "win_recon_mean"),
    ]
    for name, path, key in specs:
        d = _load_val_test(name, path, key)
        thr, vf = select_threshold(d["val"]["score"], d["val"]["y_win"])
        m, _ = _eval_window(d["test"]["y_win"], d["test"]["score"], thr,
                            d["test"]["y_point"], win=12,
                            point_offset=None)
        rows.append({
            "model": name, "variant": "val_best_f1",
            "level": "window12", "threshold": float(thr),
            "val_F1": float(vf), **m,
            "n": len(d["test"]["score"]),
            "positives": int(d["test"]["y_win"].sum()),
        })

    # ---------------- 2) AT：验证集阈值 + 纯重构误差 ----------------
    at_dir = ROOT / "repro_anomaly_transformer/runs/swat_20_80"
    te = np.load(at_dir / "test_energy.npy")
    ts = pd.read_csv(at_dir / "test_scores.csv")
    yp = ts["label"].to_numpy(np.int8)[:len(te)]
    va = np.load(at_dir / "val_energy.npy")
    data = load_prepared(ROOT / "repro_anomaly_transformer/data/swat")
    test_y = data["test_y"]
    n_val = int(len(test_y) * 0.2)
    val_y = test_y[:n_val].astype(int)
    # AT 的 val_energy 是 89,900 点（非重叠 100 步窗口展平）
    val_y = val_y[:len(va)]
    w = 12
    va_w = np.lib.stride_tricks.sliding_window_view(va, w).mean(axis=1)
    val_y_w = np.lib.stride_tricks.sliding_window_view(val_y, w).max(axis=1)
    te_w = np.lib.stride_tricks.sliding_window_view(te, w).mean(axis=1)
    yp_w = np.lib.stride_tricks.sliding_window_view(yp, w).max(axis=1)
    thr, vf = select_threshold(va_w, val_y_w)
    m, _ = _eval_window(yp_w, te_w, thr, yp, win=w, point_offset=w - 1)
    rows.append({
        "model": "AnomalyTransformer(paper)", "variant": "val_best_f1",
        "level": "window12", "threshold": float(thr),
        "val_F1": float(vf), **m,
        "n": len(te_w), "positives": int(yp_w.sum()),
    })

    # AT 纯重构误差（用同一个 checkpoint 重新推理 val/test）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = data["train_X"].shape[1]
    model = AnomalyTransformer(win_size=100, enc_in=n_channels,
                               c_out=n_channels, e_layers=3).to(device)
    model.load_state_dict(torch.load(
        at_dir / "swat_checkpoint.pth", map_location=device,
        weights_only=True))
    model.eval()

    def _loss_points(X, y):
        ds = WindowDataset(X, y, win_size=100, step=100, mode="thre")
        loader = DataLoader(ds, batch_size=32, shuffle=False,
                            num_workers=0, pin_memory=True)
        out = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device, non_blocking=True)
                rec, _, _, _ = model(xb)
                loss = ((xb - rec) ** 2).mean(dim=-1)
                out.append(loss.cpu().numpy())
        return np.concatenate(out, axis=0).reshape(-1)

    val_X = data["test_X"][:n_val]
    test_X = data["test_X"][n_val:]
    val_loss = _loss_points(val_X, val_y)
    test_loss = _loss_points(test_X, test_y[n_val:].astype(int))
    nv = min(len(val_loss), len(val_y))
    val_loss, val_y2 = val_loss[:nv], val_y[:nv]
    val_lw = np.lib.stride_tricks.sliding_window_view(val_loss, w).mean(axis=1)
    val_yw = np.lib.stride_tricks.sliding_window_view(val_y2, w).max(axis=1)
    test_lw = np.lib.stride_tricks.sliding_window_view(test_loss, w).mean(axis=1)
    test_yp = test_y[n_val:].astype(int)[:len(test_loss)]
    test_yw = np.lib.stride_tricks.sliding_window_view(test_yp, w).max(axis=1)
    thr, vf = select_threshold(val_lw, val_yw)
    m, _ = _eval_window(test_yw, test_lw, thr, test_yp, win=w,
                        point_offset=w - 1)
    rows.append({
        "model": "AnomalyTransformer(paper)",
        "variant": "reconstruction_loss + val_best_f1",
        "level": "window12", "threshold": float(thr),
        "val_F1": float(vf), **m,
        "n": len(test_lw), "positives": int(test_yw.sum()),
    })

    # ---------------- 3) SBT：验证集上检查分数方向 ----------------
    sbt_dir = ROOT / "repro_anomaly_transformer/runs/sbt_20_80"
    sbt_ckpt = sbt_dir / "sbt_checkpoint.pth"
    if sbt_ckpt.exists():
        sbt = build_sbt(n_channels, 60).to(device)
        sbt.load_state_dict(torch.load(sbt_ckpt, map_location=device,
                                      weights_only=True))
        sbt.eval()
        val_X60 = data["test_X"][:n_val]
        val_y60 = data["test_y"][:n_val].astype(int)
        ds = WindowDataset(val_X60, val_y60, win_size=60, step=1, mode="val")
        loader = DataLoader(ds, batch_size=128, shuffle=False,
                            num_workers=0, pin_memory=True)
        vals = []
        with torch.no_grad():
            for xb, _ in loader:
                out = sbt.score_batch(xb.to(device, non_blocking=True))
                vals.append(out["window"]["score_max"].cpu().numpy())
        val_s = np.concatenate(vals).reshape(-1)
        val_yw = np.lib.stride_tricks.sliding_window_view(
            val_y60, 60).max(axis=1)
        n = min(len(val_s), len(val_yw))
        val_s, val_yw = val_s[:n], val_yw[:n]
        test_df = pd.read_csv(sbt_dir / "test_scores.csv")
        test_s = test_df["score"].to_numpy(np.float64)
        test_yw = test_df["label_window"].to_numpy(int)
        test_yp = data["test_y"][n_val:].astype(int)

        candidates = []
        for sign in (+1, -1):
            vs = sign * val_s
            thr, vf = select_threshold(vs, val_yw)
            auc = (float(roc_auc_score(val_yw, vs))
                   if len(np.unique(val_yw)) > 1 else np.nan)
            candidates.append((sign, thr, vf, auc))
        sign, thr, vf, auc = max(candidates, key=lambda x: x[2])
        ts = sign * test_s
        m, _ = _eval_window(test_yw, ts, thr, test_yp, win=60,
                            point_offset=59)
        rows.append({
            "model": "SBT",
            "variant": f"val_selected_sign={sign}",
            "level": "window60", "threshold": float(thr),
            "val_F1": float(vf), **m,
            "n": len(test_s), "positives": int(test_yw.sum()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    cols = ["model", "variant", "level", "threshold", "val_F1",
            "AUC", "PR_AUC", "Precision", "Recall", "F1", "FPR",
            "PA_F1", "n", "positives"]
    lines = ["# 快速改进实验（SWaT，20/80 冻结测试）", "",
             "- 训练：Normal only；验证：Attack 前 20%；测试：后 80% 冻结",
             "- `val_best_f1`：阈值只在验证集上选",
             "- SBT 的 sign 也只在验证集上选",
             "- `PA_F1`：窗口预测映射到最后点后做 point-adjust", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, x in df.iterrows():
        vals = []
        for c in cols:
            v = x[c]
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

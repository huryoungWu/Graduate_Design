# -*- coding: utf-8 -*-
"""SBT 的 20/80 复现：Normal 训练，Attack 前 20% 验证，后 80% 冻结测试。

SBT 只输出窗口分数（score_max/score_mean），因此评估单位是 60 步窗口；
阈值在验证集上按 F1 选，测试集只在最后评估。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)
from torch.utils.data import DataLoader

from models.common.protocol import select_threshold
from models.reconstruction.sbt import build_model
from models.reconstruction.usad_extended_metrics import point_adjust

from .dataset import WindowDataset, load_prepared

ROOT = Path(__file__).resolve().parents[1]


def _win_labels(y, win):
    return np.lib.stride_tricks.sliding_window_view(
        np.asarray(y).astype(np.int8), win).max(axis=1)


def main():
    p = argparse.ArgumentParser(description="SBT 20/80 SWaT")
    p.add_argument("--data-dir", default=str(
        ROOT / "repro_anomaly_transformer" / "data" / "swat"))
    p.add_argument("--out-dir", default=str(
        ROOT / "repro_anomaly_transformer" / "runs" / "sbt_20_80"))
    p.add_argument("--win-size", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-rows", type=int, default=10000)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_prepared(args.data_dir)
    train_x = data["train_X"]
    test_x = data["test_X"]
    test_y = data["test_y"]
    test_ts = data["test_timestamp"]

    if args.smoke:
        n = min(args.smoke_rows, len(train_x), len(test_x))
        train_x, test_x, test_y = train_x[:n], test_x[:n], test_y[:n]
        test_ts = test_ts[:n]

    n_val = int(len(test_x) * args.val_frac)
    val_x, val_y = test_x[:n_val], test_y[:n_val]
    test_x, test_y = test_x[n_val:], test_y[n_val:]
    test_ts = test_ts[n_val:]
    print(f"[split] train={len(train_x):,} | val={len(val_x):,} | "
          f"frozen test={len(test_x):,}", flush=True)

    win = args.win_size
    train_ds = WindowDataset(train_x, None, win_size=win, step=1, mode="train")
    val_ds = WindowDataset(val_x, val_y, win_size=win, step=1, mode="val")
    test_ds = WindowDataset(test_x, test_y, win_size=win, step=1, mode="test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)
    print(f"[windows] train={len(train_ds):,} val={len(val_ds):,} "
          f"test={len(test_ds):,}", flush=True)

    model = build_model(train_x.shape[1], win).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs))
    print(f"[SBT] params="
          f"{sum(p.numel() for p in model.parameters()):,}", flush=True)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * x.size(0)
            seen += x.size(0)
        scheduler.step()
        print(f"[SBT] Epoch {epoch:02d}/{args.epochs} | "
              f"loss={total / max(seen, 1):.6f}", flush=True)
    print(f"[SBT] train time {time.time() - t0:.1f}s", flush=True)

    torch.save(model.state_dict(), out_dir / "sbt_checkpoint.pth")

    @torch.no_grad()
    def score_loader(loader):
        model.eval()
        maxs, means = [], []
        for x, _ in loader:
            out = model.score_batch(x.to(device, non_blocking=True))
            maxs.append(out["window"]["score_max"].cpu().numpy())
            means.append(out["window"]["score_mean"].cpu().numpy())
        return (np.concatenate(maxs).reshape(-1),
                np.concatenate(means).reshape(-1))

    val_max, val_mean = score_loader(val_loader)
    test_max, test_mean = score_loader(test_loader)
    val_labels = _win_labels(val_y, win)
    test_labels = _win_labels(test_y, win)
    nv = min(len(val_max), len(val_labels))
    nt = min(len(test_max), len(test_labels))
    val_max, val_mean = val_max[:nv], val_mean[:nv]
    test_max, test_mean = test_max[:nt], test_mean[:nt]
    val_labels, test_labels = val_labels[:nv], test_labels[:nt]

    candidates = {"score_max": (val_max, test_max),
                  "score_mean": (val_mean, test_mean)}
    rows = []
    for name, (vs, ts) in candidates.items():
        thr, vf = select_threshold(vs, val_labels)
        pred = (ts > thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            test_labels, pred, average="binary", zero_division=0)
        auc = (float(roc_auc_score(test_labels, ts))
               if len(np.unique(test_labels)) > 1 else float("nan"))
        ap = (float(average_precision_score(test_labels, ts))
              if len(np.unique(test_labels)) > 1 else float("nan"))
        rows.append({
            "score": name, "threshold": float(thr), "val_F1": float(vf),
            "Precision": float(p), "Recall": float(r), "F1": float(f1),
            "FPR": float(((pred == 1) & (test_labels == 0)).sum()
                         / max((test_labels == 0).sum(), 1)),
            "AUC": auc, "PR_AUC": ap,
            "n_test_windows": int(len(ts)),
            "positives": int(test_labels.sum()),
        })
    best = max(rows, key=lambda x: x["val_F1"])
    # 选验证集 F1 最高的候选分数，测试集只评估
    ts = candidates[best["score"]][1]
    thr = best["threshold"]
    pred = (ts > thr).astype(int)

    # 把窗口预测映射到窗口最后一点，算 point-adjusted 指标（辅助）
    T = len(test_y)
    point_pred = np.zeros(T, dtype=np.int8)
    m = min(len(pred), T - win + 1)
    point_pred[win - 1:win - 1 + m] = pred[:m]
    pa_pred = point_adjust(test_y, point_pred)
    p_pa, r_pa, f1_pa, _ = precision_recall_fscore_support(
        test_y, pa_pred, average="binary", zero_division=0)

    metrics = {
        "dataset": "swat",
        "model": "SBT",
        "protocol": "20/80 frozen test (val_best_f1)",
        "train_data": "Normal only",
        "test_used_in_training": False,
        "win_size": win,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "best_score": best["score"],
        "threshold": thr,
        "val_F1": best["val_F1"],
        "Precision": best["Precision"],
        "Recall": best["Recall"],
        "F1": best["F1"],
        "FPR": best["FPR"],
        "AUC": best["AUC"],
        "PR_AUC": best["PR_AUC"],
        "PA_Precision": float(p_pa),
        "PA_Recall": float(r_pa),
        "PA_F1": float(f1_pa),
        "n_test_windows": best["n_test_windows"],
        "positives": best["positives"],
        "candidates": rows,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    import pandas as pd
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(test_ts[win - 1:win - 1 + len(ts)]).astype(str),
        "label_window": test_labels.astype(int),
        "score": ts,
        "pred": pred,
    })
    df.to_csv(out_dir / "test_scores.csv", index=False,
              encoding="utf-8-sig")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k != "candidates"}, ensure_ascii=False, indent=2))
    print(f"已保存: {out_dir}")


if __name__ == "__main__":
    main()

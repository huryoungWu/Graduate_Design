# -*- coding: utf-8 -*-
"""评估大容量 USAD / LSTM-VAE 在 Normal-99 + 20/80 下的表现。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)

from models.classical.normal99_full_attack import (_scale,
                                                   _score_window_model,
                                                   load_periods)
from models.common.config import resolve_device
from models.reconstruction.lstm_vae import LSTMVAE
from models.reconstruction.usad import UsadModel
from models.reconstruction.usad_extended_metrics import point_adjust

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "08_SWaT" / "SWaT_all_1s.csv"
OUT = ROOT / "runs" / "recon_comparison" / "COMPARISON_LARGE_CAPACITY.csv"


def _eval(artifact_dir, score_key, normal_key, model, L, scaler, device,
          clip):
    normal, attack, cols = load_periods(str(CSV))
    nx, _, _ = _scale(normal, attack, cols, scaler, clip)
    ns = _score_window_model(
        model, nx, np.zeros(len(nx), np.int8), L, device, normal_key)
    thr = float(np.percentile(ns, 99))
    with np.load(Path(artifact_dir) / "test.npz", allow_pickle=False) as z:
        s = np.asarray(z[f"win_{score_key}"], dtype=np.float64)
        y = np.asarray(z["label_window"]).astype(int)
        yp = np.asarray(z["label_point"]).astype(int)
    n = min(len(s), len(y))
    s, y = s[:n], y[:n]
    pred = (s > thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    neg = y == 0
    fpr = float(((pred == 1) & neg).sum() / max(neg.sum(), 1))
    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan
    ap = (float(average_precision_score(y, s))
          if len(np.unique(y)) > 1 else np.nan)
    pa = point_adjust(yp[:n], pred)
    pp, pr, pf1, _ = precision_recall_fscore_support(
        yp[:n], pa, average="binary", zero_division=0)
    return {
        "threshold": thr, "AUC": auc, "PR_AUC": ap,
        "Precision": float(p), "Recall": float(r), "F1": float(f1),
        "FPR": fpr, "PA_Precision": float(pp), "PA_Recall": float(pr),
        "PA_F1": float(pf1), "n": int(n), "positives": int(y.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--usad-dir", default=str(ROOT / "runs/usad_h100"))
    p.add_argument("--usad-hidden", type=int, default=100)
    p.add_argument("--lstmvae-dir", default=str(ROOT / "runs/lstmvae_large"))
    p.add_argument("--lstmvae-hidden", type=int, default=64)
    p.add_argument("--lstmvae-z", type=int, default=32)
    args = p.parse_args()
    device = resolve_device("cuda")
    L = 12
    rows = []

    if (Path(args.usad_dir) / "model.pt").exists():
        normal, attack, cols = load_periods(str(CSV))
        model = UsadModel(L, len(cols), hidden_size=args.usad_hidden,
                          clip_input=False).to(device)
        model.load_state_dict(torch.load(
            Path(args.usad_dir) / "model.pt", map_location=device,
            weights_only=True))
        m = _eval(args.usad_dir, "usad_score", "usad_score", model, L,
                  "minmax", device, False)
        rows.append({"model": f"USAD(hidden={args.usad_hidden})",
                     **m})
        del model, normal, attack
        torch.cuda.empty_cache()

    if (Path(args.lstmvae_dir) / "model.pt").exists():
        normal, attack, cols = load_periods(str(CSV))
        model = LSTMVAE(len(cols), L, intermediate_dim=args.lstmvae_hidden,
                        z_dim=args.lstmvae_z, clip_input=False).to(device)
        model.load_state_dict(torch.load(
            Path(args.lstmvae_dir) / "model.pt", map_location=device,
            weights_only=True))
        m = _eval(args.lstmvae_dir, "recon_mean", "recon_mean", model, L,
                  "minmax11", device, False)
        rows.append({"model": f"LSTM-VAE(hidden={args.lstmvae_hidden},"
                              f"z={args.lstmvae_z})", **m})

    if not rows:
        print("没有找到大容量模型的 model.pt")
        return
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"已保存: {OUT}")


if __name__ == "__main__":
    main()

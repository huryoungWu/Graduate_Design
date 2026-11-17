# -*- coding: utf-8 -*-
"""Forecasting vs reconstruction anomaly detectors on cached datasets.

Protocol (same as robust_dfa_fusion):
- train on normal windows only; input clipped to train [0.1%, 99.9%]
- forecasting score: next-step MSE (mean over channels)
- reconstruction score: reconstruction MSE at the window's last step
- scores -> 12-step mean, aligned to the shared labels
- Normal99 / ValBestF1; report P/R/F1/FPR/AUC/PA-F1

Run one dataset per process:
    python -m repro_anomaly_transformer.forecast_recon_benchmark --dataset psm
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.common.config import resolve_device
from models.forecasting.lstm import LSTMForecaster
from models.forecasting.micn import MICNForecaster
from models.forecasting.moderntcn import ModernTCNForecaster
from models.forecasting.tcn import TCNForecaster
from models.forecasting.transformer import TimeSeriesTransformer
from models.reconstruction.ae import AutoEncoder
from models.reconstruction.lstm_vae import LSTMVAE
from models.reconstruction.usad import UsadModel
from models.reconstruction.vae import VAE

from .robust_dfa_fusion import (
    L, W, _clip_train, _metrics_for, _prep_tcn, _prep_tcn_normal,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "multidataset"
OUT_DIR = ROOT / "runs" / "forecast_recon_benchmark"
EPOCHS = 10
MAX_BATCHES = 800

FORECAST = ["TCN", "ModernTCN", "MICN", "LSTM", "Transformer"]
RECON = ["AE", "VAE", "LSTMVAE", "USAD"]


class WindowDS(Dataset):
    def __init__(self, X, seq_len=L):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, i):
        return self.X[i:i + self.seq_len], self.X[i + self.seq_len]


def build(kind, n_ch):
    if kind == "TCN":
        return TCNForecaster(n_ch, L, 1)
    if kind == "ModernTCN":
        return ModernTCNForecaster(n_ch, L, 1)
    if kind == "MICN":
        return MICNForecaster(n_ch, L, 1)
    if kind == "LSTM":
        return LSTMForecaster(n_ch, L, 1)
    if kind == "Transformer":
        return TimeSeriesTransformer(n_ch, L, 1)
    if kind == "AE":
        return AutoEncoder(L, n_ch)
    if kind == "VAE":
        return VAE(L, n_ch)
    if kind == "LSTMVAE":
        return LSTMVAE(n_ch, L)
    if kind == "USAD":
        return UsadModel(L, n_ch)
    raise ValueError(kind)


def minmax_train(Xtr, *others):
    lo = Xtr.min(axis=0)
    hi = Xtr.max(axis=0)
    scale = np.maximum(hi - lo, 1e-6)
    return [np.clip((a - lo) / scale, 0.0, 1.0) for a in (Xtr, *others)]


def train_model(kind, Xtr, device, epochs=EPOCHS, max_batches=MAX_BATCHES):
    torch.manual_seed(42)
    model = build(kind, Xtr.shape[1]).to(device)
    ds = WindowDS(Xtr)
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    t0 = time.time()
    if kind == "USAD":
        # bound USAD's full-loader loop to the same batch budget
        n = min(len(ds), max_batches * 256)
        sub = torch.utils.data.Subset(ds, list(range(n)))
        loader = DataLoader(sub, batch_size=256, shuffle=True,
                            num_workers=0)
        model.fit(loader, epochs=epochs, lr=1e-4, device=device,
                  name="USAD")
        return model.eval(), time.time() - t0
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(epochs):
        model.train()
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            opt.zero_grad()
            if kind in ("AE", "VAE", "LSTMVAE"):
                loss = model.training_loss(x)
            else:
                loss = nn.functional.mse_loss(model(x)[:, 0, :], y.to(device))
            loss.backward()
            opt.step()
    return model.eval(), time.time() - t0


@torch.no_grad()
def forecast_err(model, X, device, batch=2048):
    view = np.lib.stride_tricks.sliding_window_view(X, L + 1, axis=0)
    parts = []
    for s in range(0, len(view), batch):
        w = view[s:s + batch].transpose(0, 2, 1)
        x = torch.from_numpy(
            np.ascontiguousarray(w[:, :-1, :])).float().to(device)
        y = torch.from_numpy(
            np.ascontiguousarray(w[:, -1, :])).float().to(device)
        pred = model(x)[:, 0, :]
        parts.append(((pred - y) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


@torch.no_grad()
def recon_err(kind, model, X, device, batch=1024):
    view = np.lib.stride_tricks.sliding_window_view(X, L, axis=0)
    parts = []
    for s in range(0, len(view), batch):
        w = view[s:s + batch].transpose(0, 2, 1)
        x = torch.from_numpy(np.ascontiguousarray(w)).float().to(device)
        if kind == "AE":
            recon = model(x)
        elif kind == "USAD":
            recon = model(x)
        else:
            recon = model(x)[0]
        err = ((recon[:, -1, :] - x[:, -1, :]) ** 2).mean(dim=1)
        parts.append(err.cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def run_one(ds, kind, Xtr, Xv, Xt, yv, yt, device,
            epochs=EPOCHS, max_batches=MAX_BATCHES):
    if kind == "USAD":
        Xtr_u, Xv_u, Xt_u = minmax_train(Xtr, Xv, Xt)
    else:
        Xtr_u, Xv_u, Xt_u = Xtr, Xv, Xt
    model, fit_s = train_model(kind, Xtr_u, device,
                               epochs=epochs, max_batches=max_batches)
    if kind in RECON:
        en = recon_err(kind, model, Xtr_u, device)
        ev = recon_err(kind, model, Xv_u, device)
        et = recon_err(kind, model, Xt_u, device)
    else:
        en = forecast_err(model, Xtr_u, device)
        ev = forecast_err(model, Xv_u, device)
        et = forecast_err(model, Xt_u, device)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tv, yv2 = _prep_tcn(ev, yv)
    tt, yt2 = _prep_tcn(et, yt)
    nv = min(len(tv), len(yv2))
    nt = min(len(tt), len(yt2))
    tv, yv2 = tv[:nv], yv2[:nv]
    tt, yt2 = tt[:nt], yt2[:nt]
    tn = _prep_tcn_normal(en)
    met = _metrics_for(yv2, tv, yt2, tt, tn)
    row = {"dataset": ds, "model": kind,
           "family": "forecast" if kind in FORECAST else "reconstruction",
           "score": "forecast_mse" if kind in FORECAST else "recon_mse",
           "fit_s": fit_s, **met}
    print(f"  [{kind}] {fit_s:.1f}s N99 P/R/F1="
          f"{met['Normal99_Precision']:.3f}/"
          f"{met['Normal99_Recall']:.3f}/{met['Normal99_F1']:.3f} "
          f"AUC={met['Normal99_AUC']:.3f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-batches", type=int, default=MAX_BATCHES)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    d = CACHE / args.dataset
    if not d.exists():
        raise SystemExit(f"missing cache: {d}")
    Xtr = np.load(d / "train_X.npy")
    Xv = np.load(d / "val_X.npy")
    yv = np.load(d / "val_y.npy").astype(int)
    Xt = np.load(d / "test_X.npy")
    yt = np.load(d / "test_y.npy").astype(int)
    Xtr, Xv, Xt = _clip_train(Xtr, Xv, Xt)
    device = resolve_device("auto")
    print(f"[f/r] {args.dataset} train={Xtr.shape} test={Xt.shape} "
          f"device={device}", flush=True)
    models = args.models or (FORECAST + RECON)
    rows = []
    for kind in models:
        try:
            rows.append(run_one(args.dataset, kind, Xtr, Xv, Xt,
                                yv, yt, device,
                                epochs=args.epochs,
                                max_batches=args.max_batches))
        except Exception as exc:
            print(f"  [error] {kind}: {type(exc).__name__}: {exc}",
                  flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not rows:
        print("no results")
        return
    out = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"{args.dataset}.csv", index=False,
               encoding="utf-8-sig")
    print(f"wrote {out_dir / (args.dataset + '.csv')}", flush=True)


if __name__ == "__main__":
    main()

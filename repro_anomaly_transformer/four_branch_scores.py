# -*- coding: utf-8 -*-
"""Generate branch scores for four/five-branch robust fusion.

Branches:
- raw: raw COPOD (unclipped input)
- tcn_raw: TCN latent COPOD (mean_z partner)
- tcn_whiten: TCN whitened latent COPOD (max_z partner)
- tcn_err: TCN one-step forecast MSE
- ae_err: AE last-step reconstruction MSE
- usad_err: USAD last-step reconstruction MSE (min-max input)

Each dataset is processed in its own process. Scores are aligned to the
shared 12-step evaluation grid and saved to runs/four_branch_scores/<ds>.npz
as <branch>_n / <branch>_v / <branch>_t plus y_val / y_test.
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
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from models.common.config import resolve_device
from models.forecasting.tcn import TCNForecaster
from models.reconstruction.ae import AutoEncoder
from models.reconstruction.usad import UsadModel

from .robust_dfa_fusion import (
    FIT_N, L, W, _clip_train, _copod_scores, _fit_copod, _prep_raw,
    _prep_raw_normal, _prep_tcn, _prep_tcn_normal,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "multidataset"
OUT = ROOT / "runs" / "four_branch_scores"
EPOCHS = 10
MAX_BATCHES = 800


class WindowDS(Dataset):
    def __init__(self, X, seq_len=L):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, i):
        return self.X[i:i + self.seq_len], self.X[i + self.seq_len]


def train_tcn(Xtr, device):
    torch.manual_seed(42)
    model = TCNForecaster(Xtr.shape[1], L, 1).to(device)
    loader = DataLoader(WindowDS(Xtr), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        for i, (x, y) in enumerate(loader):
            if i >= MAX_BATCHES:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x)[:, 0, :], y)
            loss.backward()
            opt.step()
    return model.eval(), time.time() - t0


@torch.no_grad()
def tcn_latent(model, X, device, batch=2048):
    view = np.lib.stride_tricks.sliding_window_view(X, L, axis=0)
    parts = []
    for s in range(0, len(view), batch):
        w = view[s:s + batch].transpose(0, 2, 1)
        t = torch.from_numpy(np.ascontiguousarray(w)).float().to(device)
        h = model.blocks(model.input_proj(t.transpose(1, 2)))
        parts.append(h[:, :, -1].cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


@torch.no_grad()
def tcn_err(model, X, device, batch=2048):
    view = np.lib.stride_tricks.sliding_window_view(X, L + 1, axis=0)
    parts = []
    for s in range(0, len(view), batch):
        w = view[s:s + batch].transpose(0, 2, 1)
        x = torch.from_numpy(np.ascontiguousarray(w[:, :-1, :])).float().to(device)
        y = torch.from_numpy(np.ascontiguousarray(w[:, -1, :])).float().to(device)
        pred = model(x)[:, 0, :]
        parts.append(((pred - y) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def train_ae(Xtr, device):
    torch.manual_seed(42)
    model = AutoEncoder(L, Xtr.shape[1]).to(device)
    loader = DataLoader(WindowDS(Xtr), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        for i, (x, _) in enumerate(loader):
            if i >= MAX_BATCHES:
                break
            x = x.to(device)
            opt.zero_grad()
            loss = model.training_loss(x)
            loss.backward()
            opt.step()
    return model.eval(), time.time() - t0


@torch.no_grad()
def recon_err(model, X, device, batch=1024):
    view = np.lib.stride_tricks.sliding_window_view(X, L, axis=0)
    parts = []
    for s in range(0, len(view), batch):
        w = view[s:s + batch].transpose(0, 2, 1)
        x = torch.from_numpy(np.ascontiguousarray(w)).float().to(device)
        recon = model(x)
        if isinstance(recon, tuple):
            recon = recon[0]
        err = ((recon[:, -1, :] - x[:, -1, :]) ** 2).mean(dim=1)
        parts.append(err.cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def minmax(Xtr, *others):
    lo = Xtr.min(0)
    hi = Xtr.max(0)
    scale = np.maximum(hi - lo, 1e-6)
    return [np.clip((a - lo) / scale, 0, 1) for a in (Xtr, *others)]


def train_usad(Xtr, device):
    torch.manual_seed(42)
    model = UsadModel(L, Xtr.shape[1]).to(device)
    n = min(len(Xtr) - L, MAX_BATCHES * 256)
    ds = torch.utils.data.Subset(WindowDS(Xtr), list(range(n)))
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    t0 = time.time()
    model.fit(loader, epochs=EPOCHS, lr=1e-4, device=device, name="USAD")
    return model.eval(), time.time() - t0


def align(name, s, y, is_raw=False):
    if is_raw:
        v, yv = _prep_raw(s["val"], y["val"])
        t, yt = _prep_raw(s["test"], y["test"])
        n = _prep_raw_normal(s["normal"])
    else:
        v, yv = _prep_tcn(s["val"], y["val"])
        t, yt = _prep_tcn(s["test"], y["test"])
        n = _prep_tcn_normal(s["normal"])
    return {"n": n, "v": v, "t": t, "yv": yv, "yt": yt}


def main():
    global EPOCHS, MAX_BATCHES
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-batches", type=int, default=MAX_BATCHES)
    args = ap.parse_args()
    EPOCHS, MAX_BATCHES = args.epochs, args.max_batches
    ds = args.dataset
    d = CACHE / ds
    if not d.exists():
        raise SystemExit(f"missing cache: {d}")
    Xtr = np.load(d / "train_X.npy")
    Xv = np.load(d / "val_X.npy")
    yv = np.load(d / "val_y.npy").astype(int)
    Xt = np.load(d / "test_X.npy")
    yt = np.load(d / "test_y.npy").astype(int)
    print(f"[4b] {ds} train={Xtr.shape} test={Xt.shape}", flush=True)
    device = resolve_device("auto")
    y = {"val": yv, "test": yt}
    raw_scores = {}

    # raw COPOD, unclipped
    m, fit_r = _fit_copod(Xtr, fit_n=None)
    raw_scores = {"normal": _copod_scores(m, Xtr),
                  "val": _copod_scores(m, Xv),
                  "test": _copod_scores(m, Xt)}
    del m
    gc.collect()
    print(f"  raw COPOD {fit_r:.1f}s", flush=True)

    Xtr_c, Xv_c, Xt_c = _clip_train(Xtr, Xv, Xt)
    Xc = {"normal": Xtr_c, "val": Xv_c, "test": Xt_c}

    # TCN latent + error
    tcn, t_tcn = train_tcn(Xtr_c, device)
    F = {k: tcn_latent(tcn, v, device) for k, v in Xc.items()}
    E = {k: tcn_err(tcn, v, device) for k, v in Xc.items()}
    del tcn
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    m_t, _ = _fit_copod(F["normal"])
    tcn_raw = {k: _copod_scores(m_t, F[k]) for k in F}
    idx = np.random.RandomState(42).choice(
        len(F["normal"]), min(FIT_N, len(F["normal"])), replace=False)
    pca = PCA(n_components=min(32, F["normal"].shape[1]), whiten=True,
              random_state=42).fit(F["normal"][idx])
    Fw = {k: pca.transform(v) for k, v in F.items()}
    lo = np.quantile(Fw["normal"], 0.001, axis=0)
    hi = np.quantile(Fw["normal"], 0.999, axis=0)
    Fw = {k: np.clip(v, lo, hi) for k, v in Fw.items()}
    m_w, _ = _fit_copod(Fw["normal"])
    tcn_wh = {k: _copod_scores(m_w, Fw[k]) for k in Fw}
    del F, Fw, m_t, m_w
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  TCN {t_tcn:.1f}s", flush=True)

    # AE
    ae, t_ae = train_ae(Xtr_c, device)
    ae_s = {k: recon_err(ae, v, device) for k, v in Xc.items()}
    del ae
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  AE {t_ae:.1f}s", flush=True)

    # USAD on min-max input
    Xu = minmax(Xtr_c, Xv_c, Xt_c)
    Xuc = {"normal": Xu[0], "val": Xu[1], "test": Xu[2]}
    usad, t_usad = train_usad(Xuc["normal"], device)
    usad_s = {k: recon_err(usad, v, device) for k, v in Xuc.items()}
    del usad
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  USAD {t_usad:.1f}s", flush=True)

    # align all branches
    branches = {
        "raw": align("raw", raw_scores, y, True),
        "tcn_raw": align("tcn_raw", tcn_raw, y, False),
        "tcn_whiten": align("tcn_whiten", tcn_wh, y, False),
        "tcn_err": align("tcn_err", E, y, False),
        "ae_err": align("ae_err", ae_s, y, False),
        "usad_err": align("usad_err", usad_s, y, False),
    }
    nv = min(len(b["v"]) for b in branches.values())
    nt = min(len(b["t"]) for b in branches.values())
    out = {}
    for k, b in branches.items():
        out[f"{k}_n"] = b["n"].astype(np.float32)
        out[f"{k}_v"] = b["v"][:nv].astype(np.float32)
        out[f"{k}_t"] = b["t"][:nt].astype(np.float32)
    out["y_val"] = yv[:nv].astype(np.int8)
    out["y_test"] = yt[:nt].astype(np.int8)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{ds}.npz", **out)
    print(f"  wrote {OUT / (ds + '.npz')} nv={nv} nt={nt}", flush=True)


if __name__ == "__main__":
    main()
